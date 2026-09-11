"""
Latent-action diagnostics for a VLA-JEPA checkpoint (one GPU, ~1 h at 64 batches).

What is measured (research/EXPERIMENT_BED.md, "diagnostics"):

  World-model channel. L1 world-model loss with the true latent tokens z versus controls:
    z_zeros       predictor gets no VLM information
    z_noise       Gaussian noise matched to z's per-channel std
    z_shuffle     z from another sample in the batch (wrong clip)
    z_tokshuffle  z tokens permuted within the sample (wrong order)
    scene_cut     true z and input states, targets from another clip
    copy_last     no predictor: repeat the last input state as the prediction
  A useful latent action gives wm_true well below z_zeros / z_shuffle; wm_true ~ z_zeros means
  the predictor ignores z (the bottleneck is dead or the targets are predictable from context).

  What z encodes. Cross-validated ridge R^2 and CCA from z (mean over tokens per transition,
  PCA) to the normalised action chunk; direction-word probe (6-way) on z and on the pre-action
  hidden state; same probes from the embodied tokens for comparison.

  Representation drift. Per-layer linear CKA between the checkpoint's VLM hidden states and
  the base VLM's on text positions (image, latent and embodied token positions excluded).

  Action head. MAE of predicted normalised actions on the same batches.

Usage:
  python research/diagnose.py --ckpt <run>/checkpoints/steps_30000_pytorch_model.pt --out results/x \
      --data_root $SCRATCH/libero --num_batches 64 --base_vlm $FAST/models/Qwen3-VL-2B-Instruct
Writes results.json and results.md into --out.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.diag_utils import (  # noqa: E402
    cca_correlations,
    kfold_linear_probe,
    kfold_ridge_r2,
    linear_cka,
    pca_reduce,
    permute_rows,
    summarize,
)
from starVLA.model.framework.base_framework import baseframework  # noqa: E402
from starVLA.model.framework.share_tools import read_mode_config  # noqa: E402
from starVLA.model.modules.regularizers.anchor_align import DIRECTION_WORDS, build_anchor_mask, direction_labels  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--data_mix", default="libero_all")
    p.add_argument("--num_batches", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_vlm", default=None, help="base Qwen3-VL path for the CKA drift measure (optional)")
    p.add_argument("--align_min_xyz_norm", type=float, default=0.0)
    p.add_argument("--pca_dim", type=int, default=64)
    p.add_argument("--no_action_mae", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides applied to the checkpoint's config.yaml, e.g. "
                        "framework.qwenvl.base_vlm=/path framework.vj2_model.base_encoder=/path")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    return p.parse_args()


def build_dataset(cfg, data_root, data_mix, seed):
    from omegaconf import OmegaConf

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data, resolve=True))
    data_cfg.data_root_dir = data_root
    data_cfg.data_mix = data_mix
    ds = get_vla_dataset(
        data_cfg=data_cfg,
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        seed=seed,
        mode="val",  # deterministic per index
    )
    return ds


def text_positions(model, terms):
    """Bool mask [B, L] of attended text positions: not image tokens, not latent/embodied tokens."""
    input_ids = terms["input_ids"]
    keep = build_anchor_mask(input_ids, terms["attention_mask"], model._special_token_ids)
    tok = model.qwen_vl_interface.processor.tokenizer
    image_ids = [tok.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>") if t in tok.get_vocab()]
    if image_ids:
        keep &= ~torch.isin(input_ids, torch.tensor(image_ids, device=input_ids.device))
    return keep


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    overrides = dict(kv.split("=", 1) for kv in args.set)
    model = baseframework.from_pretrained(args.ckpt, config_overrides=overrides)
    model = model.to(torch.bfloat16 if args.dtype == "bf16" else torch.float32).to(device).eval()
    if hasattr(model.target_encoder, "_mean"):
        model.target_encoder.register_imagenet_stats()
    cfg = model.config
    ds = build_dataset(cfg, args.data_root, args.data_mix, args.seed)
    print(f"[diag] checkpoint {args.ckpt}; dataset len {len(ds)}; encoder {model.target_encoder.encoder_type}; "
          f"states {model.num_states}; bottleneck {model.latent_bottleneck.kind}")

    base_vlm = None
    if args.base_vlm:
        from transformers import AutoModelForImageTextToText

        base_vlm = AutoModelForImageTextToText.from_pretrained(args.base_vlm, dtype=torch.bfloat16).to(device).eval()

    wm = {k: [] for k in ("z_true", "z_zeros", "z_noise", "z_shuffle", "z_shuffle_other_task", "z_batchmean", "z_globalmean", "z_tokshuffle", "scene_cut", "copy_last")}
    paired = {"shuffle_minus_true": [], "batchmean_minus_true": [], "globalmean_minus_true": []}
    per_transition_wm = []
    z_running_sum, z_running_n = None, 0
    contrib = {"z_embed_norm": [], "state_embed_norm": []}
    Z, ZRAW, EMB, PRE, ACT, MAE = [], [], [], [], [], []
    cka_sums, cka_n = None, 0

    indices = rng.choice(len(ds), size=args.num_batches * args.batch_size, replace=False)
    for b in range(args.num_batches):
        examples = [ds[int(i)] for i in indices[b * args.batch_size : (b + 1) * args.batch_size]]
        terms = model.world_model_terms(examples)
        z, inp, gt = terms["z"], terms["input_states"], terms["gt_states"]
        B = z.shape[0]

        def wm_loss(zz, ii, gg):
            return model.world_model_loss(zz, ii, gg).item()

        def wm_loss_per_sample(zz, ii, gg):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = model.vj_predictor(ii, zz).float()
            return (pr - gg.float()).abs().flatten(1).mean(1)  # [B]

        l_true_ps = wm_loss_per_sample(z, inp, gt)
        wm["z_true"].append(l_true_ps.mean().item())
        wm["z_zeros"].append(wm_loss(torch.zeros_like(z), inp, gt))
        std = z.float().std(dim=(0, 1), keepdim=True)
        wm["z_noise"].append(wm_loss((torch.randn_like(z.float()) * std).to(z.dtype), inp, gt))
        # constant-z controls: batch mean, and running dataset mean (from previous batches)
        z_bm = z.float().mean(0, keepdim=True).expand_as(z).to(z.dtype)
        l_bm_ps = wm_loss_per_sample(z_bm, inp, gt)
        wm["z_batchmean"].append(l_bm_ps.mean().item())
        paired["batchmean_minus_true"].extend((l_bm_ps - l_true_ps).cpu().tolist())
        if z_running_n > 0:
            z_gm = (z_running_sum / z_running_n).unsqueeze(0).expand_as(z).to(z.dtype)
            l_gm_ps = wm_loss_per_sample(z_gm, inp, gt)
            wm["z_globalmean"].append(l_gm_ps.mean().item())
            paired["globalmean_minus_true"].extend((l_gm_ps - l_true_ps).cpu().tolist())
        z_running_sum = z.float().sum(0) if z_running_sum is None else z_running_sum + z.float().sum(0)
        z_running_n += B
        # how much of the predictor input comes from z versus the context states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc = model.vj_predictor.action_encoder
            contrib["z_embed_norm"].append(enc(z).float().norm(dim=-1).mean().item())
            contrib["state_embed_norm"].append(model.vj_predictor.predictor_embed(inp).float().norm(dim=-1).mean().item())
            # decompose the predictor's view of z into the batch-constant part and the per-sample residual
            z_dev = (z.float() - z.float().mean(0, keepdim=True)).to(z.dtype)
            contrib.setdefault("z_embed_norm_of_batchmean", []).append(enc(z_bm).float().norm(dim=-1).mean().item())
            contrib.setdefault("z_embed_norm_of_residual", []).append((enc(z).float() - enc(z_bm).float()).norm(dim=-1).mean().item())
            contrib.setdefault("z_residual_norm_raw", []).append(z_dev.float().norm(dim=-1).mean().item())
            contrib.setdefault("z_batchmean_norm_raw", []).append(z_bm.float().norm(dim=-1).mean().item())
        if B >= 2:
            perm = torch.as_tensor(permute_rows(B, rng), device=device)
            l_sh_ps = wm_loss_per_sample(z[perm], inp, gt)
            wm["z_shuffle"].append(l_sh_ps.mean().item())
            paired["shuffle_minus_true"].extend((l_sh_ps - l_true_ps).cpu().tolist())
            langs = [ex["lang"] for ex in examples]
            other = [i for i in range(B) if langs[perm[i].item()] != langs[i]]
            if other:
                wm["z_shuffle_other_task"].append(l_sh_ps[other].mean().item())
            wm["scene_cut"].append(wm_loss(z, inp, gt[perm]))
        tperm = torch.as_tensor(rng.permutation(z.shape[1]), device=device)
        wm["z_tokshuffle"].append(wm_loss(z[:, tperm], inp, gt))
        tok = gt.shape[1] // model.num_transitions
        last = inp[:, -tok:].repeat(1, model.num_transitions, 1)
        wm["copy_last"].append(F.l1_loss(last.float(), gt.float()).item())
        # per-transition error with true z
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.vj_predictor(inp, z).float()
        per_transition_wm.append((pred - gt.float()).abs().view(B, model.num_transitions, -1).mean(dim=(0, 2)).cpu().numpy())

        # representations for the probes
        k = model.num_action_tokens_per_timestep
        Z.append(z.float().view(B, model.num_transitions, k, -1).mean(2).cpu().numpy())  # [B, S-1, H]
        ZRAW.append(terms["z_raw"].float().view(B, model.num_transitions, k, -1).mean(2).cpu().numpy())
        EMB.append(terms["embodied"].float().mean(1).cpu().numpy())
        last_hidden = terms["hidden_states"][-1]
        PRE.append(last_hidden[torch.arange(B, device=device), terms["pre_action_pos"]].float().cpu().numpy())
        ACT.append(terms["actions_target"].cpu().numpy())

        if base_vlm is not None:
            qwen_inputs = model._build_inputs([ex["image"] for ex in examples], [ex["lang"] for ex in examples], True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = base_vlm(**qwen_inputs, output_hidden_states=True, return_dict=True)
            keep = text_positions(model, terms)
            layers_s = terms["hidden_states"][1:]
            layers_t = out.hidden_states[1:]
            vals = []
            for hs, ht in zip(layers_s, layers_t):
                xs = hs[keep].float().cpu().numpy()
                xt = ht[keep].float().cpu().numpy()
                vals.append(linear_cka(xs, xt))
            vals = np.asarray(vals)
            cka_sums = vals if cka_sums is None else cka_sums + vals
            cka_n += 1

        if not args.no_action_mae:
            out = model.predict_action(
                batch_images=[ex["image"] for ex in examples],
                instructions=[ex["lang"] for ex in examples],
                state=[ex["state"] for ex in examples] if "state" in examples[0] else None,
            )
            gt_act = np.array([ex["action"] for ex in examples], dtype=np.float32)[:, -(model.future_action_window_size + 1) :]
            MAE.append(float(np.abs(out["normalized_actions"] - gt_act).mean()))
        if (b + 1) % 8 == 0:
            print(f"[diag] batch {b + 1}/{args.num_batches}  wm_true {np.mean(wm['z_true']):.4f}  z_zeros {np.mean(wm['z_zeros']):.4f}  ({time.time() - t0:.0f}s)")

    Z = np.concatenate(Z)  # [N, S-1, H]
    ZRAW = np.concatenate(ZRAW)
    EMB = np.concatenate(EMB)
    PRE = np.concatenate(PRE)
    ACT = np.concatenate(ACT)  # [N, T, D]
    N = Z.shape[0]

    results = {
        "checkpoint": args.ckpt,
        "num_samples": int(N),
        "encoder_type": model.target_encoder.encoder_type,
        "bottleneck": model.latent_bottleneck.kind,
        "num_transitions": int(model.num_transitions),
        "world_model": {k: summarize(v) for k, v in wm.items() if v},
        "world_model_per_transition_true_z": np.stack(per_transition_wm).mean(0).tolist(),
    }
    results["world_model"]["paired_per_sample"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "abs_mean": float(np.mean(np.abs(v))), "n": len(v)} for k, v in paired.items() if v
    }
    results["world_model"]["predictor_input_norms"] = {k: float(np.mean(v)) for k, v in contrib.items()}
    results["world_model"]["z_effect"] = {
        "zeros_minus_true": results["world_model"]["z_zeros"]["mean"] - results["world_model"]["z_true"]["mean"],
        "shuffle_minus_true": results["world_model"]["z_shuffle"]["mean"] - results["world_model"]["z_true"]["mean"],
        "copy_last_minus_true": results["world_model"]["copy_last"]["mean"] - results["world_model"]["z_true"]["mean"],
    }

    # z -> actions
    z_flat = Z.reshape(N, -1)
    zp, zstats = pca_reduce(z_flat, args.pca_dim)
    act_flat = ACT.reshape(N, -1)
    results["z_to_actions"] = {
        "pca_dim": int(zstats["k"]),
        "pca_evr": zstats["evr"],
        "ridge_r2_pca": kfold_ridge_r2(zp, act_flat),
        "ridge_r2_full": kfold_ridge_r2(z_flat, act_flat),
        "ridge_r2_xyz_only": kfold_ridge_r2(zp, ACT[:, :, :3].reshape(N, -1)),
        "cca_top8": cca_correlations(z_flat, act_flat, k=8, pca_dim=args.pca_dim),
        "z_token_std_mean": float(Z.std(0).mean()),
        # geometry: how large is the per-sample variation of z relative to its mean vector?
        "z_mean_vector_norm": float(np.linalg.norm(Z.mean(0), axis=-1).mean()),
        "z_deviation_norm_mean": float(np.linalg.norm(Z - Z.mean(0, keepdims=True), axis=-1).mean()),
        "z_pairwise_cosine_mean": float(_pairwise_cosine(Z.reshape(N, -1))),
        "z_raw_pairwise_cosine_mean": float(_pairwise_cosine(ZRAW.reshape(N, -1))),
        "embodied_pairwise_cosine_mean": float(_pairwise_cosine(EMB)),
        "z_effective_rank": float(np.exp(-(lambda p: (p * np.log(p + 1e-12)).sum())(
            (lambda s: s / s.sum())(np.linalg.svd(z_flat - z_flat.mean(0), compute_uv=False) ** 2)))),
    }
    results["embodied_to_actions"] = {"ridge_r2_pca": kfold_ridge_r2(pca_reduce(EMB, args.pca_dim)[0], act_flat)}
    results["pre_action_to_actions"] = {"ridge_r2_pca": kfold_ridge_r2(pca_reduce(PRE, args.pca_dim)[0], act_flat)}
    if model.latent_bottleneck.kind != "none":
        zr = ZRAW.reshape(N, -1)
        results["z_raw_to_actions"] = {"ridge_r2_pca": kfold_ridge_r2(pca_reduce(zr, args.pca_dim)[0], act_flat)}

    # direction probe
    labels = direction_labels(torch.from_numpy(ACT), min_norm=args.align_min_xyz_norm).numpy()
    results["direction_probe"] = {
        "label_hist": np.bincount(labels[labels >= 0], minlength=len(DIRECTION_WORDS)).tolist(),
        "valid_frac": float((labels >= 0).mean()),
        "from_z_pca": kfold_linear_probe(zp, labels, len(DIRECTION_WORDS)),
        "from_pre_action_pca": kfold_linear_probe(pca_reduce(PRE, args.pca_dim)[0], labels, len(DIRECTION_WORDS)),
        "from_embodied_pca": kfold_linear_probe(pca_reduce(EMB, args.pca_dim)[0], labels, len(DIRECTION_WORDS)),
    }
    if cka_n:
        results["text_cka_vs_base_vlm_per_layer"] = (cka_sums / cka_n).tolist()
        results["text_cka_vs_base_vlm_last"] = float(cka_sums[-1] / cka_n)
    if MAE:
        results["action_mae_normalized"] = summarize(MAE)
    results["seconds"] = time.time() - t0

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_markdown(results, os.path.join(args.out, "results.md"))
    print(json.dumps({k: v for k, v in results.items() if k in ("world_model", "z_to_actions", "direction_probe", "action_mae_normalized")}, indent=1))
    print(f"[diag] wrote {args.out}/results.json")


def _pairwise_cosine(x: np.ndarray, max_n: int = 512) -> float:
    x = x[:max_n].astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    c = x @ x.T
    n = c.shape[0]
    return float((c.sum() - np.trace(c)) / max(n * (n - 1), 1))


def write_markdown(r, path):
    wm = r["world_model"]
    lines = [f"# Diagnostics: `{r['checkpoint']}`", "",
             f"encoder `{r['encoder_type']}`, bottleneck `{r['bottleneck']}`, {r['num_samples']} samples, {r['num_transitions']} transitions", "",
             "## World-model L1 (lower is better)", "", "| condition | mean | std |", "|---|---|---|"]
    for k in ("z_true", "z_zeros", "z_noise", "z_shuffle", "z_shuffle_other_task", "z_batchmean", "z_globalmean", "z_tokshuffle", "scene_cut", "copy_last"):
        if k in wm and isinstance(wm[k], dict) and wm[k].get("n"):
            lines.append(f"| {k} | {wm[k]['mean']:.4f} | {wm[k]['std']:.4f} |")
    e = wm["z_effect"]
    lines += ["", f"z effect: zeros-true {e['zeros_minus_true']:+.4f}, shuffle-true {e['shuffle_minus_true']:+.4f}, copy_last-true {e['copy_last_minus_true']:+.4f}", ""]
    pp = wm.get("paired_per_sample", {})
    for k, v in pp.items():
        lines.append(f"- paired per-sample {k}: mean {v['mean']:+.4f}, |diff| mean {v['abs_mean']:.4f}, std {v['std']:.4f} (n={v['n']})")
    nm = wm.get("predictor_input_norms", {})
    if nm:
        lines.append(f"- predictor input norms: action_encoder(z) {nm['z_embed_norm']:.2f} vs predictor_embed(states) {nm['state_embed_norm']:.2f}")
        if "z_embed_norm_of_residual" in nm:
            lines.append(f"- z decomposition (raw -> after action_encoder): batch-mean part {nm['z_batchmean_norm_raw']:.1f} -> {nm['z_embed_norm_of_batchmean']:.1f}; "
                         f"per-sample residual {nm['z_residual_norm_raw']:.1f} -> {nm['z_embed_norm_of_residual']:.1f}")
    lines.append("")
    za = r["z_to_actions"]
    lines += ["## z -> action chunk", "",
              f"- ridge R^2 (PCA {za['pca_dim']}, evr {za['pca_evr']:.2f}): **{za['ridge_r2_pca']['r2']:.3f}**; full-dim {za['ridge_r2_full']['r2']:.3f}; xyz only {za['ridge_r2_xyz_only']['r2']:.3f}",
              f"- CCA top-8: {', '.join(f'{c:.2f}' for c in za['cca_top8'])}",
              f"- effective rank of z: {za['z_effective_rank']:.1f}; ||mean z|| {za['z_mean_vector_norm']:.1f} vs mean ||z - mean z|| {za['z_deviation_norm_mean']:.1f}; "
              f"pairwise cosine z {za['z_pairwise_cosine_mean']:.3f}, embodied {za['embodied_pairwise_cosine_mean']:.3f}",
              f"- embodied tokens -> actions R^2: {r['embodied_to_actions']['ridge_r2_pca']['r2']:.3f}; pre-action hidden -> actions R^2: {r['pre_action_to_actions']['ridge_r2_pca']['r2']:.3f}", ""]
    dp = r["direction_probe"]
    lines += ["## Direction probe (6-way)", "",
              f"- labels {dp['label_hist']} (valid {dp['valid_frac']:.2f})",
              f"- from z: acc {dp['from_z_pca']['acc']:.3f} (chance {dp['from_z_pca']['chance']:.3f})",
              f"- from pre-action hidden: acc {dp['from_pre_action_pca']['acc']:.3f}",
              f"- from embodied: acc {dp['from_embodied_pca']['acc']:.3f}", ""]
    if "text_cka_vs_base_vlm_per_layer" in r:
        c = r["text_cka_vs_base_vlm_per_layer"]
        lines += ["## Text-token CKA vs base VLM", "", f"- last layer {c[-1]:.3f}; per layer: {', '.join(f'{v:.2f}' for v in c)}", ""]
    if "action_mae_normalized" in r:
        lines += [f"## Action MAE (normalised): {r['action_mae_normalized']['mean']:.4f}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

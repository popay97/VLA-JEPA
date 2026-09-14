"""
Two laptop-scale measurements on a VLA-JEPA checkpoint (one GPU, ~8 GB, a few minutes).

1. Encoder leak. The clip-level V-JEPA 2 encoder is bidirectional over the 8-frame clip, so
   the "current" state s_0 (frames 0-1) can contain information about frames 2-7. We measure it
   directly: encode the clip, then re-encode with the future frames replaced (by another
   sample's frames, or frozen at frame 1) and report the relative change of every state,
       leak_k = ||s_k' - s_k|| / ||s_k - mean_batch(s_k)||
   (denominator: the per-state deviation scale, so 1.0 means "as different as another
   sample"). For the per-frame encoder (Meta's V-JEPA2-AC trick, same weights) the change of
   s_0 must be exactly zero. As a control we also perturb the PAST frames and report the change
   of the last state, which any encoder is allowed to show.

2. Gradient decomposition. d L_wm / d z is the only signal the world-model loss sends to the
   VLM. We decompose z = m + r (m: batch-mean per slot, the shared component; r: per-sample
   residual) and report, per token,
       sensitivity_m = |<g, m>|      loss change under a 100% rescale of the shared part
       sensitivity_r = |<g, r>|      loss change under a 100% rescale of the residual
       ||g||, |<g, m_hat>|, ||g - <g, m_hat> m_hat||   (gradient energy along / off the mean)
   and compare ||g|| per latent token with ||d L_action / d embodied|| per embodied token, i.e.
   the other gradient the VLM receives (weighted with the configured wm_loss_weight).
   Optionally the same with the centering bottleneck applied post hoc (untrained, for the
   loss reference only).

Usage:
  python research/leak_and_gradient.py --ckpt <...>.pt --out <dir> --data_root <libero> \
      --data_mix libero_spatial --num_batches 16 --batch_size 4 [--set KEY=VALUE ...]
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

from research.diag_utils import summarize  # noqa: E402
from research.diagnose import build_dataset  # noqa: E402
from starVLA.model.framework.base_framework import baseframework  # noqa: E402
from starVLA.model.modules.world_model.target_encoders import TargetEncoder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--data_mix", default="libero_spatial")
    p.add_argument("--num_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--skip_leak", action="store_true")
    p.add_argument("--skip_gradient", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- leak
def _rel_change(a: torch.Tensor, b: torch.Tensor, num_states: int):
    """a, b: [B, S*tok, D]. Returns per-state relative change [S] and cosine [S]."""
    B = a.shape[0]
    a = a.float().view(B, num_states, -1)
    b = b.float().view(B, num_states, -1)
    diff = (a - b).norm(dim=-1)  # [B, S]
    scale = (a - a.mean(0, keepdim=True)).norm(dim=-1).mean(0, keepdim=True).clamp(min=1e-6)  # [1, S]
    cos = F.cosine_similarity(a, b, dim=-1)  # [B, S]
    return (diff / scale).mean(0).cpu().numpy(), cos.mean(0).cpu().numpy()


@torch.no_grad()
def leak_measurement(model, videos: np.ndarray, rng, perframe: TargetEncoder):
    """videos: uint8 [B, V, T, H, W, 3]. Returns dict of per-state arrays for both encoders."""
    B, V, T = videos.shape[:3]
    tub = model.target_encoder.tubelet_size  # frames per clip state (2)
    perm = np.roll(np.arange(B), 1)
    # future swap: frames >= tub come from another sample; past swap: frames < tub from another sample
    fut_swap = videos.copy()
    fut_swap[:, :, tub:] = videos[perm][:, :, tub:]
    past_swap = videos.copy()
    past_swap[:, :, :tub] = videos[perm][:, :, :tub]
    # future frozen: hold frame tub-1 for the rest of the clip (no motion after the present)
    fut_frozen = videos.copy()
    fut_frozen[:, :, tub:] = videos[:, :, tub - 1 : tub]

    out = {}
    for name, enc, S in (("clip", model.target_encoder, model.num_states), ("perframe", perframe, perframe.num_states(T))):
        base = enc.encode(videos)
        res = {}
        for pname, pv in (("future_swap", fut_swap), ("future_frozen", fut_frozen), ("past_swap", past_swap)):
            rel, cos = _rel_change(base, enc.encode(pv), S)
            res[pname] = {"rel_change_per_state": rel.tolist(), "cosine_per_state": cos.tolist()}
        out[name] = res
    return out


# --------------------------------------------------------------------------- gradient
def gradient_measurement(model, examples, wm_weight: float):
    """Gradient of the world-model loss w.r.t. z, decomposed into shared / residual parts,
    and the action-loss gradient w.r.t. the embodied tokens for scale."""
    terms = model.world_model_terms(examples)
    z0, inp, gt = terms["z_raw"].detach(), terms["input_states"], terms["gt_states"]
    B, K, H = z0.shape

    z = z0.float().clone().requires_grad_(True)
    with torch.enable_grad():
        zb, _, _ = model.latent_bottleneck(z.to(z0.dtype))
        loss = model.world_model_loss(zb, inp, gt)
        (g,) = torch.autograd.grad(loss * wm_weight, z)
    g = g.float()  # [B, K, H]
    zf = z0.float()
    m = zf.mean(0, keepdim=True).expand_as(zf)  # shared component per slot
    r = zf - m  # residual
    m_hat = m / m.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    g_along = (g * m_hat).sum(-1)  # [B, K]
    g_off = (g - g_along.unsqueeze(-1) * m_hat).norm(dim=-1)  # [B, K]
    res = {
        "wm_loss": float(loss),
        "grad_norm_per_token": float(g.norm(dim=-1).mean()),
        "grad_along_mean_dir_abs": float(g_along.abs().mean()),
        "grad_off_mean_dir_norm": float(g_off.mean()),
        "sensitivity_shared_abs": float((g * m).sum(-1).abs().mean()),  # |<g, m>|
        "sensitivity_residual_abs": float((g * r).sum(-1).abs().mean()),  # |<g, r>|
        "z_shared_norm": float(m.norm(dim=-1).mean()),
        "z_residual_norm": float(r.norm(dim=-1).mean()),
        "cos_grad_residual_abs": float(F.cosine_similarity(g, r, dim=-1).abs().mean()),
        "cos_grad_mean_abs": float(F.cosine_similarity(g, m, dim=-1).abs().mean()),
    }
    # finite-difference check of the two sensitivities: rescale each part by 1 +/- eps
    with torch.no_grad():
        eps = 0.1

        def L(zz):
            return float(model.world_model_loss(zz.to(z0.dtype), inp, gt)) * wm_weight

        res["fd_shared_x1.1_minus_base"] = L(m * (1 + eps) + r) - float(loss) * wm_weight
        res["fd_residual_x1.1_minus_base"] = L(m + r * (1 + eps)) - float(loss) * wm_weight
        res["fd_residual_x2_minus_base"] = L(m + 2 * r) - float(loss) * wm_weight
        res["fd_residual_x0_minus_base"] = L(m) - float(loss) * wm_weight

    # action-loss gradient w.r.t. the embodied tokens, for scale
    if "actions_target" in terms:
        emb0 = terms["embodied"].detach()
        emb = emb0.clone().requires_grad_(True)  # model dtype (bf16), as in training
        state = None
        if "state" in examples[0]:
            state = torch.tensor(np.array([ex["state"] for ex in examples]), device=emb.device, dtype=emb.dtype)
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.float32):
            torch.manual_seed(0)
            a_loss = model.action_model(emb, terms["actions_target"].to(emb.dtype), state)
            (ga,) = torch.autograd.grad(a_loss, emb)
        res["action_loss"] = float(a_loss)
        res["action_grad_norm_per_embodied_token"] = float(ga.float().norm(dim=-1).mean())
        res["wm_to_action_grad_ratio"] = res["grad_norm_per_token"] / max(res["action_grad_norm_per_embodied_token"], 1e-12)
    return res


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    overrides = dict(kv.split("=", 1) for kv in args.set)
    model = baseframework.from_pretrained(args.ckpt, config_overrides=overrides)
    model = model.to(torch.bfloat16).to(device).eval()
    cfg = model.config
    ds = build_dataset(cfg, args.data_root, args.data_mix, args.seed)
    wm_weight = float(model.wm_loss_weight)
    print(f"[lg] ckpt {args.ckpt}; encoder {model.target_encoder.encoder_type}; states {model.num_states}; wm_w {wm_weight}")

    perframe = None
    if not args.skip_leak:
        from omegaconf import OmegaConf

        vj = OmegaConf.create(OmegaConf.to_container(cfg.framework.vj2_model, resolve=True))
        vj.encoder_type = "vjepa2_perframe"
        vj.state_stride = 1
        vj.normalize_targets = False
        perframe = TargetEncoder(vj, model=model.vj_encoder, processor=model.vj_processor)

    leak = {"clip": {}, "perframe": {}}
    grads = []
    indices = rng.choice(len(ds), size=args.num_batches * args.batch_size, replace=False)
    for b in range(args.num_batches):
        examples = [ds[int(i)] for i in indices[b * args.batch_size : (b + 1) * args.batch_size]]
        if not args.skip_leak:
            videos = np.stack([ex["video"] for ex in examples])
            lk = leak_measurement(model, videos, rng, perframe)
            for enc in lk:
                for pname, v in lk[enc].items():
                    d = leak[enc].setdefault(pname, {"rel": [], "cos": []})
                    d["rel"].append(v["rel_change_per_state"])
                    d["cos"].append(v["cosine_per_state"])
        if not args.skip_gradient:
            grads.append(gradient_measurement(model, examples, wm_weight))
        print(f"[lg] batch {b + 1}/{args.num_batches} ({time.time() - t0:.0f}s)")

    results = {"checkpoint": args.ckpt, "num_samples": int(args.num_batches * args.batch_size), "wm_loss_weight": wm_weight}
    if not args.skip_leak:
        results["encoder_leak"] = {
            enc: {p: {"rel_change_per_state": np.mean(v["rel"], 0).tolist(), "cosine_per_state": np.mean(v["cos"], 0).tolist()} for p, v in d.items()}
            for enc, d in leak.items()
        }
    if grads:
        results["gradient"] = {k: summarize([g[k] for g in grads]) for k in grads[0]}
    results["seconds"] = time.time() - t0
    with open(os.path.join(args.out, "leak_gradient.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_markdown(results, os.path.join(args.out, "leak_gradient.md"))
    print(json.dumps(results, indent=1))


def write_markdown(r, path):
    lines = [f"# Encoder leak and gradient decomposition: `{r['checkpoint']}`", "", f"{r['num_samples']} samples, wm_loss_weight {r['wm_loss_weight']}", ""]
    if "encoder_leak" in r:
        lines += ["## Encoder leak: relative change of each state when frames are perturbed", "",
                  "rel = ||s_k' - s_k|| / (per-state deviation scale); 0 = unchanged, 1 = as different as another sample.", ""]
        for enc, d in r["encoder_leak"].items():
            lines += [f"### `{enc}` encoder", "", "| perturbation | " + " | ".join(f"s_{k}" for k in range(len(next(iter(d.values()))["rel_change_per_state"]))) + " |",
                      "|---|" + "---|" * len(next(iter(d.values()))["rel_change_per_state"])]
            for p, v in d.items():
                lines.append(f"| {p} rel | " + " | ".join(f"{x:.3f}" for x in v["rel_change_per_state"]) + " |")
                lines.append(f"| {p} cos | " + " | ".join(f"{x:.4f}" for x in v["cosine_per_state"]) + " |")
            lines.append("")
    if "gradient" in r:
        lines += ["## Gradient of the (weighted) world-model loss w.r.t. z", "", "| quantity | mean | std |", "|---|---|---|"]
        for k, v in r["gradient"].items():
            lines.append(f"| {k} | {v['mean']:.6g} | {v['std']:.3g} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

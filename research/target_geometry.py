"""
Encoder-only measurements on the world-model *targets* (no VLM loaded, fits an 8 GB laptop):

  1. Leak. Re-encode each clip with the frames after the present replaced (another sample's
     frames, or frozen at the last present frame) and with the past frames replaced, and report
     the relative change of every state. Leak-free encoders must show exactly 0 for the present
     state under future perturbations.
  2. Geometry. For the raw per-view token features x in R^D:
        mean-vector norm ||E x||, deviation norm E||x - E x||, energy fraction of the mean
        direction ||E x||^2 / E||x||^2, pairwise cosine between tokens of different samples,
        variance fraction of the first principal component after centering, and the number of
        "massive" channels (|mean_c| > 10 * median_c std_c).
     Reported for the raw features, after per-token LayerNorm (`normalize_targets`), and after
     dataset-mean centering followed by LayerNorm. A large mean-direction energy fraction means
     the L1 target is dominated by a constant that any predictor learns as a bias.
  3. Scale. Mean |s_{k+1} - s_k| (what copy-last leaves) versus E|x - E x| (what a mean
     predictor leaves), to calibrate world-model losses across encoders.

Usage:
  python research/target_geometry.py --ckpt <...>.pt --data_root <libero> --out <dir> \
      --encoders clip perframe levjepa --vjepa2 facebook/vjepa2-vitl-fpc64-256 \
      --levjepa galilai-group/LeVJEPA-VideoMix-Large
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

from research.diagnose import build_dataset  # noqa: E402
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config  # noqa: E402
from starVLA.model.modules.world_model.target_encoders import TargetEncoder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint whose config.yaml defines the dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--data_mix", default="libero_spatial")
    p.add_argument("--num_batches", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoders", nargs="+", default=["clip", "perframe", "levjepa"])
    p.add_argument("--vjepa2", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--levjepa", default="galilai-group/LeVJEPA-VideoMix-Large")
    p.add_argument("--tokens_per_batch", type=int, default=1024, help="tokens subsampled per batch for PCA / cosine")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    return p.parse_args()


class NS(dict):
    """dict with attribute access and .get, enough for TargetEncoder's config reads."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def build_encoders(names, args, device, dtype):
    encs = {}
    vj_model = vj_proc = None
    for name in names:
        if name in ("clip", "perframe"):
            cfg = NS(encoder_type="vjepa2_clip" if name == "clip" else "vjepa2_perframe", base_encoder=args.vjepa2,
                     normalize_targets=False, state_stride=1, num_frames=8)
            enc = TargetEncoder(cfg, model=vj_model, processor=vj_proc)
            if vj_model is None:
                enc.model.to(device, dtype)
                vj_model, vj_proc = enc.model, enc.processor
        elif name == "levjepa":
            cfg = NS(encoder_type="levjepa", base_encoder=args.levjepa, normalize_targets=False, state_stride=1, num_frames=8)
            enc = TargetEncoder(cfg)
            enc.model.to(device, dtype)
            enc.register_imagenet_stats()
        else:
            raise ValueError(name)
        encs[name] = enc
    return encs


# --------------------------------------------------------------------------- leak
def rel_change(a, b, S):
    B = a.shape[0]
    a = a.float().view(B, S, -1)
    b = b.float().view(B, S, -1)
    diff = (a - b).norm(dim=-1)
    scale = (a - a.mean(0, keepdim=True)).norm(dim=-1).mean(0, keepdim=True).clamp(min=1e-6)
    cos = F.cosine_similarity(a, b, dim=-1)
    return (diff / scale).mean(0).cpu().numpy(), cos.mean(0).cpu().numpy()


@torch.no_grad()
def leak(enc, videos, present_frames):
    """present_frames: number of frames that make up the present state(s) we protect (2 for the
    clip encoder's tubelet, 2 also for the others so the comparison is like for like)."""
    B, V, T = videos.shape[:3]
    perm = np.roll(np.arange(B), 1)
    fut_swap = videos.copy(); fut_swap[:, :, present_frames:] = videos[perm][:, :, present_frames:]
    fut_frozen = videos.copy(); fut_frozen[:, :, present_frames:] = videos[:, :, present_frames - 1 : present_frames]
    past_swap = videos.copy(); past_swap[:, :, :present_frames] = videos[perm][:, :, :present_frames]
    S = enc.num_states(T)
    base = enc.encode(videos)
    out = {}
    for pname, pv in (("future_swap", fut_swap), ("future_frozen", fut_frozen), ("past_swap", past_swap)):
        rel, cos = rel_change(base, enc.encode(pv), S)
        out[pname] = (rel, cos)
    return base, out


# --------------------------------------------------------------------------- geometry
class GeometryAccumulator:
    def __init__(self, D, tokens_per_batch, rng):
        self.D = D
        self.sum = torch.zeros(D, dtype=torch.float64)
        self.sumsq = torch.zeros(D, dtype=torch.float64)
        self.sum_norm2 = 0.0
        self.n = 0
        self.samples = []  # subsampled tokens with a sample id for cross-sample cosine
        self.sample_ids = []
        self.k = tokens_per_batch
        self.rng = rng
        self.copy_last_l1 = []
        self.abs_dev_l1 = []

    def add(self, states, S, V):
        """states: [B, S*tok, V*D] -> per-view tokens [B, S, tok, V, D]."""
        B = states.shape[0]
        x = states.float().view(B, S, -1, V, self.D)
        flat = x.permute(0, 3, 1, 2, 4).reshape(B * V, S, -1, self.D)  # [B*V, S, tok, D]
        # copy-last vs mean-deviation scale (L1, as the world-model loss)
        self.copy_last_l1.append((flat[:, 1:] - flat[:, :-1]).abs().mean().item())
        toks = flat.reshape(-1, self.D).double().cpu()
        self.sum += toks.sum(0)
        self.sumsq += toks.pow(2).sum(0)
        self.sum_norm2 += toks.pow(2).sum().item()
        self.n += toks.shape[0]
        ids = torch.arange(B * V).view(-1, 1, 1).expand(B * V, S, flat.shape[2]).reshape(-1)
        idx = torch.from_numpy(self.rng.choice(toks.shape[0], size=min(self.k, toks.shape[0]), replace=False))
        self.samples.append(toks[idx].float())
        self.sample_ids.append(ids[idx])

    def _stats(self, X, ids, mean, per_channel_std):
        """X: [N, D] float tokens (already transformed); mean: [D] of the transformed distribution."""
        N = X.shape[0]
        mean_norm = mean.norm().item()
        dev = (X - mean).norm(dim=-1).mean().item()
        energy_frac = (mean.pow(2).sum() / X.pow(2).sum(-1).mean()).item()
        # cross-sample pairwise cosine
        Xn = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        C = Xn @ Xn.T
        diff_mask = ids[:, None] != ids[None, :]
        cos_cross = C[diff_mask].mean().item()
        # PCA of centered tokens
        Xc = X - X.mean(0, keepdim=True)
        s = torch.linalg.svdvals(Xc)
        var = s.pow(2)
        pc1_frac = (var[0] / var.sum()).item()
        eff_rank = torch.exp(-(var / var.sum() * torch.log(var / var.sum() + 1e-12)).sum()).item()
        massive = int((mean.abs() > 10 * per_channel_std.median()).sum().item())
        top_channel = int(mean.abs().argmax().item())
        return {
            "mean_vector_norm": mean_norm,
            "deviation_norm_mean": dev,
            "mean_to_deviation_ratio": mean_norm / max(dev, 1e-8),
            "energy_fraction_of_mean_direction": energy_frac,
            "pairwise_cosine_cross_sample": cos_cross,
            "pc1_variance_fraction_after_centering": pc1_frac,
            "effective_rank_after_centering": eff_rank,
            "massive_channels": massive,
            "top_channel": top_channel,
            "top_channel_abs_mean_over_median_std": (mean.abs().max() / per_channel_std.median()).item(),
        }

    def finalize(self):
        X = torch.cat(self.samples)
        ids = torch.cat(self.sample_ids)
        mean = (self.sum / self.n).float()
        std = ((self.sumsq / self.n) - (self.sum / self.n).pow(2)).clamp(min=0).sqrt().float()
        out = {"num_tokens_total": int(self.n), "num_tokens_sampled": int(X.shape[0])}
        out["raw"] = self._stats(X, ids, mean, std)
        Xln = F.layer_norm(X, (self.D,))
        out["after_layernorm"] = self._stats(Xln, ids, Xln.mean(0), Xln.std(0))
        Xcl = F.layer_norm(X - mean, (self.D,))
        out["after_centering_then_layernorm"] = self._stats(Xcl, ids, Xcl.mean(0), Xcl.std(0))
        # what an L1 predictor faces: copy-last residual vs mean-deviation residual (raw scale)
        out["scale_raw"] = {
            "copy_last_l1": float(np.mean(self.copy_last_l1)),
            "mean_abs_deviation_l1": float((X - mean).abs().mean()),
            "mean_abs_value_l1": float(X.abs().mean()),
        }
        return out


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    model_cfg, _ = read_mode_config(args.ckpt)
    cfg = dict_to_namespace(model_cfg)
    ds = build_dataset(cfg, args.data_root, args.data_mix, args.seed)
    encs = build_encoders(args.encoders, args, device, dtype)
    for n, e in encs.items():
        print(f"[tg] {n}: type {e.encoder_type} img {e.img_size} grid {e.grid} D {e.hidden_size} states(8) {e.num_states(8)}")

    leak_acc = {n: {} for n in encs}
    geo = {}
    indices = rng.choice(len(ds), size=args.num_batches * args.batch_size, replace=False)
    for b in range(args.num_batches):
        examples = [ds[int(i)] for i in indices[b * args.batch_size : (b + 1) * args.batch_size]]
        videos = np.stack([ex["video"] for ex in examples])  # [B, V, T, H, W, 3]
        V, T = videos.shape[1], videos.shape[2]
        for n, enc in encs.items():
            base, lk = leak(enc, videos, present_frames=2)
            for pname, (rel, cos) in lk.items():
                d = leak_acc[n].setdefault(pname, {"rel": [], "cos": []})
                d["rel"].append(rel); d["cos"].append(cos)
            if n not in geo:
                geo[n] = GeometryAccumulator(enc.hidden_size, args.tokens_per_batch, rng)
            geo[n].add(base, enc.num_states(T), V)
        print(f"[tg] batch {b + 1}/{args.num_batches} ({time.time() - t0:.0f}s)")

    results = {"num_samples": int(args.num_batches * args.batch_size), "data_mix": args.data_mix, "encoders": {}}
    for n, enc in encs.items():
        results["encoders"][n] = {
            "encoder_type": enc.encoder_type,
            "num_states_for_8_frames": enc.num_states(8),
            "leak": {p: {"rel_change_per_state": np.mean(v["rel"], 0).tolist(), "cosine_per_state": np.mean(v["cos"], 0).tolist()} for p, v in leak_acc[n].items()},
            "geometry": geo[n].finalize(),
        }
        # per-channel dataset mean of the raw per-view tokens, for `vj2_model.center_targets_path`
        mean_path = os.path.join(args.out, f"target_mean_{n}_{args.data_mix}.pt")
        torch.save({"mean": (geo[n].sum / geo[n].n).float().cpu(), "encoder_type": enc.encoder_type, "data_mix": args.data_mix,
                    "num_tokens": int(geo[n].n), "num_samples": int(args.num_batches * args.batch_size)}, mean_path)
        results["encoders"][n]["mean_path"] = mean_path
    results["seconds"] = time.time() - t0
    with open(os.path.join(args.out, "target_geometry.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_markdown(results, os.path.join(args.out, "target_geometry.md"))
    print(json.dumps({n: r["geometry"] for n, r in results["encoders"].items()}, indent=1))


def write_markdown(r, path):
    L = [f"# Target-encoder leak and geometry ({r['data_mix']}, {r['num_samples']} samples)", ""]
    L += ["## Leak: relative change of each state (0 = unchanged, ~1.4 = as different as another sample)", ""]
    for n, e in r["encoders"].items():
        S = e["num_states_for_8_frames"]
        L += [f"### `{n}` ({e['encoder_type']}, {S} states)", "", "| perturbation | " + " | ".join(f"s_{k}" for k in range(S)) + " |", "|---|" + "---|" * S]
        for p, v in e["leak"].items():
            L.append(f"| {p} rel | " + " | ".join(f"{x:.3f}" for x in v["rel_change_per_state"]) + " |")
        L.append("")
    L += ["## Geometry of the target tokens", ""]
    keys = ["mean_vector_norm", "deviation_norm_mean", "mean_to_deviation_ratio", "energy_fraction_of_mean_direction",
            "pairwise_cosine_cross_sample", "pc1_variance_fraction_after_centering", "effective_rank_after_centering",
            "massive_channels", "top_channel_abs_mean_over_median_std"]
    for stage in ("raw", "after_layernorm", "after_centering_then_layernorm"):
        L += [f"### {stage}", "", "| quantity | " + " | ".join(f"`{n}`" for n in r["encoders"]) + " |", "|---|" + "---|" * len(r["encoders"])]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(f"{r['encoders'][n]['geometry'][stage][k]:.4g}" for n in r["encoders"]) + " |")
        L.append("")
    L += ["### scale (raw, L1)", "", "| quantity | " + " | ".join(f"`{n}`" for n in r["encoders"]) + " |", "|---|" + "---|" * len(r["encoders"])]
    for k in ("copy_last_l1", "mean_abs_deviation_l1", "mean_abs_value_l1"):
        L.append(f"| {k} | " + " | ".join(f"{r['encoders'][n]['geometry']['scale_raw'][k]:.4g}" for n in r["encoders"]) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

"""
Statistics for the Anchor-Align direction labels on LIBERO (T2 in research/TASKS.md).

Reads the LeRobot parquet files directly (no model), computes the chunk-averaged xyz action
deltas in the *normalised* units the training batch sees (min-max to [-1, 1] from the
dataset's own q01/q99 statistics as in the GR00T loader), and prints:
  * per-axis mean (the zero point: LIBERO deltas are not centred on 0 after min-max)
  * percentiles of the centred xyz norm -> pick `align_min_xyz_norm`
  * label histogram at a few thresholds -> check the 6 classes are all populated

Usage:
  python research/align_stats.py --data_root $SCRATCH/libero --chunk 7 [--datasets name1 name2]
Then set framework.anchor_align.align_min_xyz_norm / align_zero_point in the arm yaml.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starVLA.model.modules.regularizers.anchor_align import DIRECTION_WORDS  # noqa: E402

DEFAULT_DATASETS = [
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
]


def load_actions(dataset_dir: str, action_key: str = "action"):
    import pandas as pd

    files = sorted(glob.glob(os.path.join(dataset_dir, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset_dir}/data")
    chunks, episodes = [], []
    for f in files:
        df = pd.read_parquet(f, columns=[action_key, "episode_index"])
        chunks.append(np.stack(df[action_key].to_numpy()))
        episodes.append(df["episode_index"].to_numpy())
    return np.concatenate(chunks).astype(np.float32), np.concatenate(episodes)


def chunk_means(actions, episodes, chunk):
    """Mean over `chunk` consecutive steps within an episode (edge-padded like the loader)."""
    out = []
    for ep in np.unique(episodes):
        a = actions[episodes == ep]
        n = len(a)
        pad = np.concatenate([a, np.repeat(a[-1:], chunk, axis=0)])
        cs = np.cumsum(np.concatenate([np.zeros((1, a.shape[1])), pad]), axis=0)
        out.append((cs[chunk : chunk + n] - cs[:n]) / chunk)
    return np.concatenate(out)


def minmax_normalize(x, q01, q99):
    return np.clip(2.0 * (x - q01) / np.maximum(q99 - q01, 1e-8) - 1.0, -1.0, 1.0)


def labels_from_xyz(xyz, min_norm):
    norm = np.linalg.norm(xyz, axis=1)
    axis = np.abs(xyz).argmax(1)
    val = xyz[np.arange(len(xyz)), axis]
    lab = axis * 2 + (val < 0).astype(int)
    lab[norm < min_norm] = -1
    return lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--chunk", type=int, default=7)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    all_xyz_norm = []
    per_ds = {}
    for name in args.datasets:
        d = os.path.join(args.data_root, name)
        actions, episodes = load_actions(d)
        q01, q99 = np.percentile(actions, 1, axis=0), np.percentile(actions, 99, axis=0)
        cm = chunk_means(actions, episodes, args.chunk)
        xyz = minmax_normalize(cm[:, :3], q01[:3], q99[:3])
        per_ds[name] = {"n": int(len(xyz)), "xyz_mean_normalized": xyz.mean(0).tolist(), "q01": q01.tolist(), "q99": q99.tolist()}
        all_xyz_norm.append(xyz)
        print(f"{name}: {len(xyz)} steps, normalised xyz mean {xyz.mean(0).round(3)}, raw xyz mean {cm[:, :3].mean(0).round(4)}")

    xyz = np.concatenate(all_xyz_norm)
    zero_point = xyz.mean(0)
    centred = xyz - zero_point
    norm = np.linalg.norm(centred, axis=1)
    pct = {str(q): float(np.percentile(norm, q)) for q in (10, 25, 50, 75, 90)}
    print(f"\nzero point (normalised xyz mean): {zero_point.round(4).tolist()}")
    print(f"centred xyz-norm percentiles: {json.dumps(pct)}")
    hist = {}
    for thr in (0.0, pct["10"], pct["25"], pct["50"]):
        lab = labels_from_xyz(centred, thr)
        h = np.bincount(lab[lab >= 0], minlength=6)
        hist[f"{thr:.4f}"] = {"valid_frac": float((lab >= 0).mean()), "hist": dict(zip(DIRECTION_WORDS, h.tolist()))}
        print(f"min_norm {thr:.4f}: valid {hist[f'{thr:.4f}']['valid_frac']:.2f}  {hist[f'{thr:.4f}']['hist']}")
    rec = {"align_zero_point": zero_point.tolist(), "recommended_align_min_xyz_norm": pct["25"],
           "percentiles": pct, "label_hist_by_threshold": hist, "per_dataset": per_ds, "chunk": args.chunk}
    print("\nrecommendation: align_zero_point =", np.round(zero_point, 4).tolist(), " align_min_xyz_norm =", round(pct["25"], 4))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

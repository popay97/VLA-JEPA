"""
Collect LIBERO success rates and diagnostics into one markdown table.

  python research/collect_results.py --results $WORK/results --out $WORK/results/summary.md

Layout expected: <results>/<run>/libero/<suite>/eval.log ("Total success rate: 0.xx" from
examples/LIBERO/eval_libero.py) and <results>/<run>/diagnostics/results.json (research/diagnose.py).
"""
import argparse
import glob
import json
import os
import re

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def libero_rates(run_dir):
    rates = {}
    for s in SUITES:
        log = os.path.join(run_dir, "libero", s, "eval.log")
        if not os.path.exists(log):
            continue
        txt = open(log, errors="ignore").read()
        m = re.findall(r"Total success rate:\s*([0-9.]+)", txt)
        if m:
            rates[s] = float(m[-1])
    return rates


def diag_summary(run_dir):
    p = os.path.join(run_dir, "diagnostics", "results.json")
    if not os.path.exists(p):
        return {}
    r = json.load(open(p))
    wm = r.get("world_model", {})
    return {
        "wm_true": wm.get("z_true", {}).get("mean"),
        "wm_zeros": wm.get("z_zeros", {}).get("mean"),
        "wm_shuffle": wm.get("z_shuffle", {}).get("mean"),
        "r2": r.get("z_to_actions", {}).get("ridge_r2_pca", {}).get("r2"),
        "dir_acc": r.get("direction_probe", {}).get("from_z_pca", {}).get("acc"),
        "cka_last": r.get("text_cka_vs_base_vlm_last"),
        "mae": r.get("action_mae_normalized", {}).get("mean"),
    }


def fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(a.results, "*"))):
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        rates = libero_rates(run_dir)
        d = diag_summary(run_dir)
        avg = sum(rates.values()) / len(rates) if len(rates) == len(SUITES) else None
        rows.append((name, rates, avg, d))
    hdr = "| run | spatial | object | goal | 10 | avg | wm_true | wm_zeros | wm_shuffle | z->act R2 | dir acc | CKA last | MAE |"
    sep = "|" + "---|" * 13
    lines = ["# Results", "", hdr, sep]
    for name, rates, avg, d in rows:
        lines.append(
            f"| {name} | " + " | ".join(fmt(rates.get(s)) for s in SUITES) + f" | {fmt(avg)} | "
            + " | ".join(fmt(d.get(k), 4) for k in ("wm_true", "wm_zeros", "wm_shuffle")) + " | "
            + " | ".join(fmt(d.get(k)) for k in ("r2", "dir_acc", "cka_last", "mae")) + " |"
        )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

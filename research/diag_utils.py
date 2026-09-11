"""
Pure-numpy/torch statistics used by research/diagnose.py. No model code here so the tests
can run on CPU without the training stack.

    pca_reduce            centre + project to the top-d principal components (train stats)
    kfold_ridge_r2        cross-validated ridge regression R^2 (dual form, N << D friendly)
    cca_correlations      top-k canonical correlations after PCA whitening
    linear_cka            linear centred kernel alignment between two representations
    kfold_linear_probe    cross-validated accuracy of a one-hot ridge classifier
    permute_rows          batch permutation with no fixed points (for shuffle controls)
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _center(x: np.ndarray, mean: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(0, keepdims=True) if mean is None else mean
    return x - mean, mean


def pca_reduce(x: np.ndarray, d: int, fit_on: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """x [N, D] -> [N, d'] with d' = min(d, rank). Returns (projected, stats) with the explained
    variance ratio in stats["evr"]."""
    ref = x if fit_on is None else fit_on
    ref_c, mean = _center(ref.astype(np.float64))
    # economy SVD on the smaller side
    u, s, vt = np.linalg.svd(ref_c, full_matrices=False)
    k = int(min(d, (s > 1e-8 * s.max()).sum())) if s.size else 0
    comps = vt[:k]  # [k, D]
    proj = (x.astype(np.float64) - mean) @ comps.T
    evr = float((s[:k] ** 2).sum() / max((s**2).sum(), 1e-12)) if k else 0.0
    return proj, {"mean": mean, "components": comps, "evr": evr, "k": k}


def _ridge_fit_predict(xtr, ytr, xte, alpha):
    """Dual-form ridge: works for D >> N. Returns predictions for xte."""
    xtr_c, mx = _center(xtr)
    ytr_c, my = _center(ytr)
    n = xtr_c.shape[0]
    K = xtr_c @ xtr_c.T
    A = np.linalg.solve(K + alpha * np.eye(n), ytr_c)  # [n, T]
    return (xte - mx) @ xtr_c.T @ A + my


def kfold_ridge_r2(
    x: np.ndarray, y: np.ndarray, alphas: Sequence[float] = (1.0, 10.0, 100.0, 1000.0), folds: int = 5, seed: int = 0
) -> Dict[str, float]:
    """Cross-validated R^2 (variance-weighted over outputs) of ridge x -> y, best alpha.
    x [N, D], y [N, T]. Returns {"r2", "alpha", "r2_per_alpha": {...}}."""
    x = x.reshape(x.shape[0], -1).astype(np.float64)
    y = y.reshape(y.shape[0], -1).astype(np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold_ids = np.array_split(idx, folds)
    out = {}
    for a in alphas:
        ss_res, ss_tot = 0.0, 0.0
        for f in range(folds):
            te = fold_ids[f]
            tr = np.concatenate([fold_ids[g] for g in range(folds) if g != f])
            pred = _ridge_fit_predict(x[tr], y[tr], x[te], a)
            ss_res += ((y[te] - pred) ** 2).sum()
            ss_tot += ((y[te] - y[tr].mean(0)) ** 2).sum()
        out[float(a)] = 1.0 - ss_res / max(ss_tot, 1e-12)
    best = max(out, key=out.get)
    return {"r2": float(out[best]), "alpha": float(best), "r2_per_alpha": {str(k): float(v) for k, v in out.items()}}


def cca_correlations(x: np.ndarray, y: np.ndarray, k: int = 8, pca_dim: int = 64, reg: float = 1e-3) -> List[float]:
    """Top-k canonical correlations between x [N, Dx] and y [N, Dy] after PCA to pca_dim."""
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    px, _ = pca_reduce(x, pca_dim)
    py, _ = pca_reduce(y, pca_dim)
    px, _ = _center(px)
    py, _ = _center(py)
    n = px.shape[0]
    cxx = px.T @ px / (n - 1) + reg * np.eye(px.shape[1])
    cyy = py.T @ py / (n - 1) + reg * np.eye(py.shape[1])
    cxy = px.T @ py / (n - 1)

    def inv_sqrt(c):
        w, v = np.linalg.eigh(c)
        w = np.clip(w, 1e-10, None)
        return v @ np.diag(w**-0.5) @ v.T

    m = inv_sqrt(cxx) @ cxy @ inv_sqrt(cyy)
    s = np.linalg.svd(m, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return [float(v) for v in s[:k]]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between x [N, Dx] and y [N, Dy]."""
    x = x.reshape(x.shape[0], -1).astype(np.float64)
    y = y.reshape(y.shape[0], -1).astype(np.float64)
    x, _ = _center(x)
    y, _ = _center(y)
    hsic = np.linalg.norm(x.T @ y, "fro") ** 2
    nx = np.linalg.norm(x.T @ x, "fro")
    ny = np.linalg.norm(y.T @ y, "fro")
    return float(hsic / max(nx * ny, 1e-12))


def kfold_linear_probe(
    x: np.ndarray, labels: np.ndarray, num_classes: int, alpha: float = 10.0, folds: int = 5, seed: int = 0
) -> Dict[str, float]:
    """Accuracy of a one-hot ridge classifier (argmax of ridge regression onto one-hot labels)
    under k-fold CV; labels < 0 are ignored. Returns accuracy, chance (majority) rate, n."""
    keep = labels >= 0
    x = x.reshape(x.shape[0], -1).astype(np.float64)[keep]
    lab = labels[keep].astype(int)
    n = x.shape[0]
    if n < folds * 2:
        return {"acc": float("nan"), "chance": float("nan"), "n": int(n)}
    onehot = np.eye(num_classes)[lab]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    fold_ids = np.array_split(idx, folds)
    correct = 0
    for f in range(folds):
        te = fold_ids[f]
        tr = np.concatenate([fold_ids[g] for g in range(folds) if g != f])
        pred = _ridge_fit_predict(x[tr], onehot[tr], x[te], alpha).argmax(1)
        correct += int((pred == lab[te]).sum())
    counts = np.bincount(lab, minlength=num_classes)
    return {"acc": correct / n, "chance": float(counts.max() / n), "n": int(n)}


def permute_rows(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of range(n) with no fixed points (derangement) for n >= 2."""
    if n < 2:
        return np.arange(n)
    for _ in range(100):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    return np.roll(np.arange(n), 1)


def summarize(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}

import numpy as np

from research.diag_utils import (
    cca_correlations,
    kfold_linear_probe,
    kfold_ridge_r2,
    linear_cka,
    pca_reduce,
    permute_rows,
    summarize,
)


def test_ridge_r2_recovers_linear_map_and_rejects_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 40))
    w = rng.normal(size=(40, 6))
    y = x @ w + 0.05 * rng.normal(size=(300, 6))
    good = kfold_ridge_r2(x, y)
    assert good["r2"] > 0.95
    bad = kfold_ridge_r2(x, rng.normal(size=(300, 6)))
    assert bad["r2"] < 0.1


def test_ridge_dual_form_handles_wide_inputs():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 5000))  # D >> N
    y = x[:, :3] @ rng.normal(size=(3, 2))
    r = kfold_ridge_r2(x, y, alphas=(1.0,))
    assert np.isfinite(r["r2"])


def test_cka_identity_and_invariances():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 16))
    assert abs(linear_cka(x, x) - 1.0) < 1e-6
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    assert abs(linear_cka(x, 3.0 * x @ q + 1.0) - 1.0) < 1e-6  # orthogonal transform, scale, shift
    assert linear_cka(x, rng.normal(size=(100, 16))) < 0.3


def test_cca_top_correlation_high_for_shared_signal():
    rng = np.random.default_rng(3)
    s = rng.normal(size=(400, 4))
    x = np.concatenate([s, rng.normal(size=(400, 20))], 1) @ rng.normal(size=(24, 24))
    y = np.concatenate([s, rng.normal(size=(400, 10))], 1) @ rng.normal(size=(14, 14))
    c = cca_correlations(x, y, k=6, pca_dim=24)
    assert len(c) == 6 and c[0] > 0.9 and c[3] > 0.8 and c[5] < 0.6
    assert all(0.0 <= v <= 1.0 for v in c)


def test_linear_probe_separable_and_ignores_negative_labels():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 6, size=240)
    centers = rng.normal(size=(6, 12)) * 4
    x = centers[labels] + rng.normal(size=(240, 12))
    labels[:20] = -1
    r = kfold_linear_probe(x, labels, 6)
    assert r["n"] == 220 and r["acc"] > 0.9 and r["chance"] < 0.4


def test_pca_reduce_and_permutation_helpers():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(50, 30)) @ np.diag(np.linspace(5, 0.1, 30))
    p, st = pca_reduce(x, 8)
    assert p.shape == (50, 8) and 0.5 < st["evr"] <= 1.0
    for n in (2, 3, 8, 33):
        perm = permute_rows(n, rng)
        assert sorted(perm.tolist()) == list(range(n)) and not np.any(perm == np.arange(n))
    s = summarize([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["n"] == 3

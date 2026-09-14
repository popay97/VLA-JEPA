"""Tests for the `center` / `center_ln` latent bottleneck kinds on a synthetic constant-dominated z."""
import torch

from starVLA.model.modules.world_model.latent_bottleneck import LatentActionBottleneck, build_latent_bottleneck


def _constant_dominated(n=6, k=4, h=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    const = torch.randn(k, h, generator=g) * 50.0  # per-slot shared component (huge)
    resid = torch.randn(n, k, h, generator=g)  # per-sample message (small)
    return const, resid, const.unsqueeze(0) + resid


def test_center_removes_shared_component_and_keeps_residual():
    const, resid, z = _constant_dominated()
    m = LatentActionBottleneck(dim=32, kind="center", center_num_tokens=4, center_momentum=0.9)
    m.train()
    out, losses, metrics = m(z)
    assert losses == {}
    # the first batch initialises the EMA exactly, so the output is the residual around the batch mean
    expected = z - z.mean(0, keepdim=True)
    assert torch.allclose(out, expected, atol=1e-4)
    assert metrics["metric/center_mean_norm"] > 100
    # the shared component is gone: the output norm is that of the residual, not of the constant
    assert out.norm(dim=-1).mean() < 2 * resid.norm(dim=-1).mean()
    assert bool(m.ema_initialized)


def test_center_ema_updates_in_train_only_and_is_stop_gradient():
    _, _, z = _constant_dominated()
    m = LatentActionBottleneck(dim=32, kind="center", center_num_tokens=4, center_momentum=0.5)
    m.train()
    m(z)
    mu1 = m.ema_mean.clone()
    z2 = z + 10.0
    m(z2)
    mu2 = m.ema_mean.clone()
    assert torch.allclose(mu2, 0.5 * mu1 + 0.5 * z2.mean(0), atol=1e-4)
    m.eval()
    m(z2 + 100.0)
    assert torch.equal(m.ema_mean, mu2), "eval must not touch the running mean"
    # gradient reaches z with an identity Jacobian (the mean is detached)
    zz = z.clone().requires_grad_(True)
    out, _, _ = m(zz)
    out.sum().backward()
    assert torch.allclose(zz.grad, torch.ones_like(zz))


def test_center_ln_has_unit_variance_times_gain_and_learnable_gain():
    _, _, z = _constant_dominated()
    m = LatentActionBottleneck(dim=32, kind="center_ln", center_num_tokens=4, center_gain=3.0, center_learnable_gain=True)
    m.train()
    out, _, _ = m(z)
    assert isinstance(m.gain, torch.nn.Parameter)
    assert torch.allclose(out.std(dim=-1, unbiased=False).mean(), torch.tensor(3.0), atol=0.2)
    out.sum().backward()
    assert m.gain.grad is not None


def test_center_shared_mean_and_slot_count_change():
    _, _, z = _constant_dominated()
    m = LatentActionBottleneck(dim=32, kind="center", center_per_slot=False)
    m.train()
    out, _, _ = m(z)
    assert m.ema_mean.shape == (1, 32)
    assert torch.allclose(out, z - z.mean(dim=(0, 1), keepdim=True), atol=1e-4)
    # a different number of tokens re-initialises a per-slot buffer instead of crashing
    m2 = LatentActionBottleneck(dim=32, kind="center", center_num_tokens=24)
    m2.train()
    out2, _, _ = m2(z)
    assert m2.ema_mean.shape == (4, 32) and out2.shape == z.shape


def test_center_state_dict_roundtrip_with_other_slot_count():
    _, _, z = _constant_dominated()
    src = LatentActionBottleneck(dim=32, kind="center", center_num_tokens=4)
    src.train()
    src(z)
    dst = LatentActionBottleneck(dim=32, kind="center", center_num_tokens=24)  # default K, wrong shape
    dst.load_state_dict(src.state_dict())
    assert torch.equal(dst.ema_mean, src.ema_mean) and bool(dst.ema_initialized)
    dst.eval()
    src.eval()
    a, _, _ = dst(z)
    b, _, _ = src(z)
    assert torch.equal(a, b)


def test_center_bf16_input_keeps_dtype():
    _, _, z = _constant_dominated()
    m = LatentActionBottleneck(dim=32, kind="center", center_num_tokens=4)
    m.eval()
    out, _, _ = m(z.to(torch.bfloat16))
    assert out.dtype == torch.bfloat16 and out.shape == z.shape


def test_build_center_from_config():
    m = build_latent_bottleneck({"bottleneck": "center_ln", "center_momentum": 0.95, "center_num_tokens": 56, "center_gain": 2.0}, dim=16)
    assert m.kind == "center_ln" and m.center_momentum == 0.95 and m.ema_mean.shape == (56, 16) and float(m.gain) == 2.0

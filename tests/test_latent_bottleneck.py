import pytest
import torch

from starVLA.model.modules.world_model.latent_bottleneck import KINDS, LatentActionBottleneck, build_latent_bottleneck


@pytest.mark.parametrize("kind", KINDS)
def test_shape_preserved(kind):
    m = LatentActionBottleneck(dim=16, kind=kind, bottleneck_dim=4, vq_codebook_size=8)
    z = torch.randn(3, 5, 16)
    out, losses, metrics = m(z)
    assert out.shape == z.shape
    assert out.dtype == z.dtype
    for v in losses.values():
        assert v.ndim == 0 and torch.isfinite(v)


def test_none_is_identity_and_drop_is_zero():
    z = torch.randn(2, 4, 8)
    out, losses, _ = LatentActionBottleneck(8, "none")(z)
    assert torch.equal(out, z) and losses == {}
    out, losses, _ = LatentActionBottleneck(8, "drop")(z)
    assert torch.count_nonzero(out) == 0 and losses == {}


def test_lowrank_gradient_flows_and_rank():
    m = LatentActionBottleneck(dim=32, kind="lowrank", bottleneck_dim=2)
    z = torch.randn(4, 3, 32, requires_grad=True)
    out, _, _ = m(z)
    out.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    # rank of the map is at most bottleneck_dim
    flat = out.detach().reshape(-1, 32) - m.up.bias.detach()
    assert torch.linalg.matrix_rank(flat, atol=1e-4) <= 2


def test_vib_kl_loss_and_eval_is_deterministic():
    m = LatentActionBottleneck(dim=16, kind="vib", bottleneck_dim=4, vib_beta=0.5)
    z = torch.randn(2, 3, 16)
    m.train()
    a, losses, metrics = m(z)
    assert "vib_kl_loss" in losses and "metric/vib_kl_raw" in metrics
    assert torch.isclose(losses["vib_kl_loss"], 0.5 * metrics["metric/vib_kl_raw"])
    m.eval()
    b1, _, _ = m(z)
    b2, _, _ = m(z)
    assert torch.equal(b1, b2)


def test_vq_perplexity_and_straight_through():
    m = LatentActionBottleneck(dim=16, kind="vq", bottleneck_dim=4, vq_codebook_size=8)
    z = torch.randn(6, 4, 16, requires_grad=True)
    out, losses, metrics = m(z)
    assert 1.0 <= metrics["metric/vq_perplexity"].item() <= 8.0 + 1e-4
    (out.sum() + losses["vq_loss"]).backward()
    assert z.grad is not None and m.codebook.weight.grad is not None


def test_noise_only_in_training():
    m = LatentActionBottleneck(dim=8, kind="layernorm", noise_std=1.0)
    z = torch.randn(2, 2, 8)
    m.eval()
    a, _, _ = m(z)
    b, _, _ = m(z)
    assert torch.equal(a, b)
    m.train()
    c, _, _ = m(z)
    assert not torch.equal(a, c)


def test_build_from_config_dict():
    m = build_latent_bottleneck({"bottleneck": "lowrank", "bottleneck_dim": 8}, dim=16)
    assert m.kind == "lowrank" and m.bottleneck_dim == 8
    assert build_latent_bottleneck(None, dim=16).kind == "none"
    with pytest.raises(ValueError):
        LatentActionBottleneck(8, "bogus")

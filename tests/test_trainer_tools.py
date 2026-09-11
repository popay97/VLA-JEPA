import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, sum_losses


class Toy(nn.Module):
    NEW_MODULE_PREFIXES = ["latent_bottleneck.", "anchor_align."]

    def __init__(self, with_new=True):
        super().__init__()
        self.qwen_vl_interface = nn.Linear(4, 4)
        self.vj_predictor = nn.Linear(4, 2)
        self.vj_encoder = nn.Linear(4, 4)
        for p in self.vj_encoder.parameters():
            p.requires_grad_(False)
        if with_new:
            self.latent_bottleneck = nn.Linear(4, 4)
            self.anchor_align = nn.Linear(4, 4)


def test_sum_losses_skips_metric_keys():
    d = {"action_loss": torch.tensor(1.0), "wm_loss": torch.tensor(0.5), "metric/foo": torch.tensor(100.0)}
    assert sum_losses(d).item() == 1.5


def test_param_groups_skip_frozen_and_route_named_modules():
    m = Toy()
    cfg = OmegaConf.create({"trainer": {"learning_rate": {"base": 1e-4, "vj_predictor": 5e-4, "missing_module": 1e-3}}})
    groups = build_param_lr_groups(m, cfg)
    ids = {id(p) for g in groups for p in g["params"]}
    assert all(id(p) not in ids for p in m.vj_encoder.parameters())  # requires_grad=False excluded
    by_name = {g["name"]: g for g in groups}
    assert by_name["vj_predictor"]["lr"] == 5e-4 and len(by_name["vj_predictor"]["params"]) == 2
    n_total_trainable = sum(1 for p in m.parameters() if p.requires_grad)
    assert sum(len(g["params"]) for g in groups) == n_total_trainable


def test_full_load_tolerates_declared_new_modules(tmp_path):
    src = Toy(with_new=False)
    ckpt = tmp_path / "old.pt"
    torch.save(src.state_dict(), ckpt)
    dst = Toy(with_new=True)
    before = dst.latent_bottleneck.weight.clone()
    TrainerUtils.load_pretrained_backbones(dst, str(ckpt))
    assert torch.equal(dst.vj_predictor.weight, src.vj_predictor.weight)
    assert torch.equal(dst.latent_bottleneck.weight, before)  # kept at init


def test_full_load_rejects_unexpected_or_missing_core_keys(tmp_path):
    src = Toy(with_new=True)
    sd = src.state_dict()
    sd["extra.weight"] = torch.zeros(1)
    torch.save(sd, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="unexpected"):
        TrainerUtils.load_pretrained_backbones(Toy(), str(tmp_path / "bad.pt"))
    sd = src.state_dict()
    del sd["vj_predictor.weight"]
    torch.save(sd, tmp_path / "bad2.pt")
    with pytest.raises(RuntimeError, match="missing"):
        TrainerUtils.load_pretrained_backbones(Toy(), str(tmp_path / "bad2.pt"))


def test_skip_load_modules_keeps_fresh_init(tmp_path):
    src = Toy()
    torch.save(src.state_dict(), tmp_path / "full.pt")
    dst = Toy()
    fresh = dst.vj_predictor.weight.clone()
    TrainerUtils.load_pretrained_backbones(dst, str(tmp_path / "full.pt"), skip_load_modules="vj_predictor")
    assert torch.equal(dst.vj_predictor.weight, fresh)
    assert torch.equal(dst.qwen_vl_interface.weight, src.qwen_vl_interface.weight)

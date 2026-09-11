"""Resumable checkpointing on CPU with a real Accelerator (no DeepSpeed): save mid-run, rebuild
everything, resume, and check step counter, LR, model weights and the stop/save triggers."""
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_scheduler

from starVLA.training.trainer_utils.checkpointing import ResumableCheckpointing, filter_state_dict


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(3, 1)
        self.anchor_align = nn.Module()
        self.anchor_align.anchor_teacher = nn.Linear(3, 1)

    def milestone_exclude_prefixes(self):
        return ["anchor_align.anchor_teacher."]

    def forward(self, x):
        return self.core(x)


class Host(ResumableCheckpointing):
    def __init__(self, out_dir, accelerator, cfg_overrides=None):
        cfg = {"output_dir": out_dir, "trainer": {"save_every_minutes": 0, "keep_last_states": 2, "auto_resume": True}}
        if cfg_overrides:
            cfg["trainer"].update(cfg_overrides)
        self.config = OmegaConf.create(cfg)
        self.accelerator = accelerator
        torch.manual_seed(0)
        model = Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        self.lr_scheduler = get_scheduler("linear", opt, num_warmup_steps=5, num_training_steps=20)
        ds = TensorDataset(torch.randn(40, 3), torch.randn(40, 1))
        dl = DataLoader(ds, batch_size=4)
        self.model, self.optimizer, self.dl = accelerator.prepare(model, opt, dl)
        self.completed_steps = 0
        self.vla_epoch_count = 0
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.init_resumable(dataloaders=[self.dl], epoch_counter_names=["vla_epoch_count"])

    def step(self, batch):
        x, y = batch
        loss = ((self.model(x) - y) ** 2).mean()
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self.completed_steps += 1


def make_accelerator():
    return Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(use_stateful_dataloader=True))


def test_filter_state_dict():
    sd = {"core.weight": 1, "anchor_align.anchor_teacher.weight": 2, "anchor_align.align_dir_proj.weight": 3}
    out = filter_state_dict(sd, ["anchor_align.anchor_teacher."])
    assert set(out) == {"core.weight", "anchor_align.align_dir_proj.weight"}


def test_save_resume_roundtrip_and_rotation(tmp_path):
    acc = make_accelerator()
    h = Host(str(tmp_path), acc)
    assert h.maybe_resume() is False
    it = iter(h.dl)
    for _ in range(3):
        h.step(next(it))
    h.save_training_state("interval")
    for _ in range(3):
        h.step(next(it))
    lr_after_6 = h.lr_scheduler.get_last_lr()[0]
    w_after_6 = h.accelerator.unwrap_model(h.model).core.weight.detach().clone()
    h.save_training_state("interval")
    for _ in range(2):
        h.step(next(it))
    h.save_training_state("interval")  # third save -> rotation keeps 2
    kept = sorted(os.listdir(os.path.join(h.state_root)))
    assert "latest" in kept and "step_3" not in kept and "step_6" in kept and "step_8" in kept
    # milestone export strips the teacher
    p = h.export_weights()
    sd = torch.load(p)
    assert "core.weight" in sd and not any(k.startswith("anchor_align.anchor_teacher.") for k in sd)
    assert os.path.exists(os.path.join(str(tmp_path), "summary.jsonl"))

    # point latest at step_6 and resume into a fresh host
    with open(h.latest_pointer, "w") as f:
        f.write("step_6")
    # a fresh Accelerator, as a requeued process would have (one registered model/optimizer)
    h2 = Host(str(tmp_path), make_accelerator())
    assert h2.maybe_resume() is True
    assert h2.completed_steps == 6
    assert abs(h2.lr_scheduler.get_last_lr()[0] - lr_after_6) < 1e-12
    assert torch.allclose(h2.accelerator.unwrap_model(h2.model).core.weight, w_after_6)
    # stateful dataloader resumes mid-epoch: the next batch is the 7th, not the 1st
    nxt = next(iter(h2.dl))[0]
    assert torch.allclose(nxt, h.dl.dataset.tensors[0][24:28])


def test_stop_triggers_sentinel_signal_and_timer(tmp_path):
    acc = make_accelerator()
    h = Host(str(tmp_path), acc, {"save_every_minutes": 1})
    assert h.check_stop_and_save() == (False, False)
    h._last_save_time -= 120
    assert h.check_stop_and_save() == (True, False)  # timer
    h._last_save_time += 120
    open(h.sentinel_path, "w").close()
    assert h.check_stop_and_save() == (True, True)  # sentinel
    h.save_training_state("stop")
    assert not os.path.exists(h.sentinel_path)  # consumed by the save
    h._stop_requested = True
    assert h.check_stop_and_save() == (True, True)  # signal flag
    h.mark_done()
    assert os.path.exists(h.done_path)


def test_explicit_resume_path_must_exist(tmp_path):
    acc = make_accelerator()
    import pytest

    with pytest.raises(FileNotFoundError):
        Host(str(tmp_path), acc, {"resume_from_checkpoint": str(tmp_path / "nope")}).maybe_resume()

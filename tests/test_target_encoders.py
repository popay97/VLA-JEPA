"""Shape/layout tests for TargetEncoder with fake encoder models (no HF downloads)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from starVLA.model.modules.world_model.target_encoders import TargetEncoder


class FakeVJEPA2(nn.Module):
    """Mimics HF VJEPA2Model.get_vision_features: [N, T, C, H, W] -> [N, (T//2)*tok, D].
    Feature value encodes (clip index, temporal step) so layout can be checked."""

    def __init__(self, img=32, patch=16, D=6):
        super().__init__()
        self.config = SimpleNamespace(image_size=img, patch_size=patch, hidden_size=D, tubelet_size=2)
        self.dummy = nn.Parameter(torch.zeros(1))
        self.D = D
        self.tok = (img // patch) ** 2

    def get_vision_features(self, pixel_values_videos):
        N, T = pixel_values_videos.shape[:2]
        S = T // 2
        # value = mean pixel of the tubelet (so per-frame duplication returns that frame's mean)
        vals = pixel_values_videos.float().view(N, S, 2, -1).mean(dim=(2, 3))  # [N, S]
        feats = vals[:, :, None, None].expand(N, S, self.tok, self.D)
        return feats.reshape(N, S * self.tok, self.D)


class FakeProcessor:
    def __call__(self, videos, return_tensors="pt"):
        if isinstance(videos, list):
            arr = np.stack(videos)
        else:
            arr = videos[None]
        return {"pixel_values_videos": torch.from_numpy(arr.astype(np.float32))}


class FakeLeVJEPA(nn.Module):
    def __init__(self, img=32, patch=16, D=6):
        super().__init__()
        self.config = SimpleNamespace(img_size=img, patch_size=patch, embed_dim=D, tubelet_size=1, attn_mode="block_causal")
        self.dummy = nn.Parameter(torch.zeros(1))
        self.D, self.tok = D, (img // patch) ** 2

    def forward(self, pixel_values):
        N, C, T, H, W = pixel_values.shape
        vals = pixel_values.mean(dim=(1, 3, 4))  # [N, T]
        patches = vals[:, :, None, None].expand(N, T, self.tok, self.D).reshape(N, T * self.tok, self.D)
        cls = torch.full((N, 1, self.D), 999.0)
        return SimpleNamespace(last_hidden_state=torch.cat([cls, patches], dim=1))


def _videos(B=2, V=2, T=8, H=32, W=32):
    # pixel value = 10*b + 100*v + t so features are traceable
    vid = np.zeros((B, V, T, H, W, 3), dtype=np.uint8)
    for b in range(B):
        for v in range(V):
            for t in range(T):
                vid[b, v, t] = 10 * b + 100 * v + t
    return vid


def _cfg(**kw):
    d = {"encoder_type": "vjepa2_clip", "base_encoder": "fake", "num_frames": 8}
    d.update(kw)
    return d


def test_vjepa2_clip_layout():
    enc = TargetEncoder(_cfg(), model=FakeVJEPA2(), processor=FakeProcessor())
    assert enc.num_states() == 4 and enc.tokens_per_state == 4 and enc.grid == 2
    out = enc.encode(_videos())
    B, V, D, tok = 2, 2, 6, 4
    assert out.shape == (B, 4 * tok, V * D)
    # state s of batch b, view v should equal mean pixel of frames (2s, 2s+1) = 10b+100v+2s+0.5
    states = out.view(B, 4, tok, V, D)
    for b in range(B):
        for v in range(V):
            for s in range(4):
                assert torch.allclose(states[b, s, :, v, :], torch.full((tok, D), 10 * b + 100 * v + 2 * s + 0.5))


def test_vjepa2_perframe_layout_and_stride():
    enc = TargetEncoder(_cfg(encoder_type="vjepa2_perframe"), model=FakeVJEPA2(), processor=FakeProcessor())
    assert enc.num_states() == 8
    out = enc.encode(_videos())
    states = out.view(2, 8, 4, 2, 6)
    for t in range(8):
        assert torch.allclose(states[1, t, :, 0, :], torch.full((4, 6), 10.0 + t))
    enc2 = TargetEncoder(_cfg(encoder_type="vjepa2_perframe", state_stride=2), model=FakeVJEPA2(), processor=FakeProcessor())
    assert enc2.num_states() == 4
    out2 = enc2.encode(_videos())
    states2 = out2.view(2, 4, 4, 2, 6)
    for s in range(4):
        assert torch.allclose(states2[0, s, :, 1, :], torch.full((4, 6), 100.0 + 2 * s))


def test_levjepa_layout_drops_cls_and_normalises():
    enc = TargetEncoder(_cfg(encoder_type="levjepa"), model=FakeLeVJEPA(), processor=None)
    assert enc.num_states() == 8 and enc.grid == 2
    out = enc.encode(_videos(H=64, W=64))  # resized to 32
    assert out.shape == (2, 8 * 4, 2 * 6)
    states = out.view(2, 8, 4, 2, 6)
    assert not torch.any(states == 999.0)  # CLS dropped
    # ImageNet-normalised means are monotone in t for a fixed (b, v)
    seq = states[0, :, 0, 0, 0]
    assert torch.all(seq[1:] > seq[:-1])


def test_normalize_targets_per_view():
    enc = TargetEncoder(_cfg(normalize_targets=True), model=FakeVJEPA2(), processor=FakeProcessor())
    out = enc.encode(_videos())
    # constant features per token -> layer norm gives zeros
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_rejects_unknown_type():
    with pytest.raises(ValueError):
        TargetEncoder(_cfg(encoder_type="bogus"), model=FakeVJEPA2(), processor=FakeProcessor())


def test_center_targets_subtracts_dataset_mean_before_layernorm(tmp_path):
    mean = torch.arange(6, dtype=torch.float32)  # per-channel dataset mean, D = 6
    p = tmp_path / "mean.pt"
    torch.save({"mean": mean}, p)
    enc = TargetEncoder(_cfg(center_targets_path=str(p)), model=FakeVJEPA2(), processor=FakeProcessor())
    ref = TargetEncoder(_cfg(), model=FakeVJEPA2(), processor=FakeProcessor())
    vid = _videos()
    out, base = enc.encode(vid), ref.encode(vid)
    assert torch.allclose(out.view(2, 4, 4, 2, 6), base.view(2, 4, 4, 2, 6) - mean, atol=1e-5)
    # with normalize_targets the constant-per-token features are no longer constant after centering
    enc2 = TargetEncoder(_cfg(center_targets_path=str(p), normalize_targets=True), model=FakeVJEPA2(), processor=FakeProcessor())
    out2 = enc2.encode(vid)
    assert out2.abs().sum() > 0 and torch.allclose(out2.view(-1, 6).mean(-1), torch.zeros(out2.numel() // 6), atol=1e-4)


def test_center_targets_rejects_wrong_channel_count(tmp_path):
    p = tmp_path / "mean.pt"
    torch.save({"mean": torch.zeros(5)}, p)
    with pytest.raises(ValueError):
        TargetEncoder(_cfg(center_targets_path=str(p)), model=FakeVJEPA2(), processor=FakeProcessor())

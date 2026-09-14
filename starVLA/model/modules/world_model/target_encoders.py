"""
Frozen target encoders for the VLA-JEPA world model.

Three ways to turn a clip of T frames per view into S latent states:

    vjepa2_clip      upstream: V-JEPA 2 on the whole clip, S = T // tubelet (2). The encoder
                     is bidirectional over the clip, so every state carries future-frame
                     information (leaks into both predictor inputs and targets).
    vjepa2_perframe  Meta's V-JEPA2-AC trick (app/vjepa_droid/train.py, forward_target):
                     each frame is duplicated into a 2-frame tubelet and encoded alone,
                     S = T // frame_stride. No temporal leakage; states are image features.
    levjepa          LeVJEPA ViT-L/16 (MLO-lab), tubelet 1, block-causal attention:
                     state t sees frames <= t only. S = T // state_stride. 224 px, 14x14 grid,
                     ImageNet normalisation, CLS token dropped.

`encode` returns states as [B, S * tokens_per_state, V * D] (views concatenated along the
feature axis, exactly the layout the predictor expects), optionally centered per view by a
precomputed per-channel dataset mean (`center_targets_path`, a .pt with key "mean" [D], produced
by `research/target_geometry.py`) and then layer-normed per view (`normalize_targets`, as in
V-JEPA2-AC). Centering matters: on LIBERO the LeVJEPA tokens put 90% of their energy into one
shared direction and LayerNorm alone does not remove it (RESULTS.md, 14 Sep 2026).
`revision` pins the HF checkpoint commit.
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_TYPES = ("vjepa2_clip", "vjepa2_perframe", "levjepa")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            v = cfg.get(key, default)
            return default if v is None else v
        except Exception:
            pass
    return getattr(cfg, key, default)


class TargetEncoder:
    """Thin, non-Module wrapper. The HF model itself is registered on the framework as
    `vj_encoder` so state-dict keys stay compatible with upstream checkpoints."""

    def __init__(self, vj_cfg, model: Optional[nn.Module] = None, processor=None):
        self.encoder_type = _cfg_get(vj_cfg, "encoder_type", "vjepa2_clip")
        if self.encoder_type not in ENCODER_TYPES:
            raise ValueError(f"unknown encoder_type {self.encoder_type!r}; choose from {ENCODER_TYPES}")
        self.path = _cfg_get(vj_cfg, "base_encoder")
        self.normalize_targets = bool(_cfg_get(vj_cfg, "normalize_targets", False))
        self.state_stride = int(_cfg_get(vj_cfg, "state_stride", 1))
        self.num_frames = int(_cfg_get(vj_cfg, "num_frames", 8))
        self.revision = _cfg_get(vj_cfg, "revision", None)
        self.center_targets_path = _cfg_get(vj_cfg, "center_targets_path", None)
        self.target_mean: Optional[torch.Tensor] = None
        if self.center_targets_path:
            blob = torch.load(self.center_targets_path, map_location="cpu")
            self.target_mean = (blob["mean"] if isinstance(blob, dict) else blob).float().flatten()

        if model is None:
            model, processor = self._load(self.path)
        self.model = model
        self.processor = processor
        if self.target_mean is not None:
            D = int(getattr(model.config, "hidden_size", getattr(model.config, "embed_dim", self.target_mean.numel())))
            if self.target_mean.numel() != D:
                raise ValueError(f"center_targets_path mean has {self.target_mean.numel()} channels, encoder has {D}")

        if self.encoder_type.startswith("vjepa2"):
            c = model.config
            self.img_size = int(c.image_size if hasattr(c, "image_size") else c.crop_size)
            self.patch_size = int(c.patch_size)
            self.hidden_size = int(c.hidden_size)
            self.tubelet_size = int(c.tubelet_size)
        else:
            c = model.config
            self.img_size = int(c.img_size)
            self.patch_size = int(c.patch_size)
            self.hidden_size = int(c.embed_dim)
            self.tubelet_size = int(c.tubelet_size)  # 1
            self.register_imagenet_stats()

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    # ------------------------------------------------------------------ loading
    def _load(self, path: str):
        from transformers import AutoModel

        kw = {"revision": self.revision} if self.revision else {}
        if self.encoder_type.startswith("vjepa2"):
            from transformers import AutoVideoProcessor

            model = AutoModel.from_pretrained(path, **kw)
            processor = AutoVideoProcessor.from_pretrained(path, **kw)
            return model, processor
        model = AutoModel.from_pretrained(path, trust_remote_code=True, **kw)
        if getattr(model.config, "attn_mode", "block_causal") != "block_causal":
            raise ValueError("LeVJEPA weights were trained block-causal; refusing to run with attn_mode="
                             f"{model.config.attn_mode!r}")
        return model, None

    def register_imagenet_stats(self):
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)

    # ------------------------------------------------------------------ geometry
    @property
    def grid(self) -> int:
        return self.img_size // self.patch_size

    @property
    def tokens_per_state(self) -> int:
        return self.grid * self.grid

    def num_states(self, num_frames: Optional[int] = None) -> int:
        T = self.num_frames if num_frames is None else num_frames
        if self.encoder_type == "vjepa2_clip":
            return T // self.tubelet_size
        return T // self.state_stride

    @property
    def device(self):
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------ encoding
    @torch.no_grad()
    def encode(self, videos: np.ndarray) -> torch.Tensor:
        """
        videos: uint8 array [B, V, T, H, W, 3] (dataset layout).
        Returns float tensor [B, S * tokens_per_state, V * D].
        """
        B, V, T, H, W, C = videos.shape
        clips = videos.reshape(B * V, T, H, W, C)
        if self.encoder_type == "vjepa2_clip":
            feats = self._encode_vjepa2_clip(clips)  # [B*V, S, tok, D]
        elif self.encoder_type == "vjepa2_perframe":
            feats = self._encode_vjepa2_perframe(clips)
        else:
            feats = self._encode_levjepa(clips)
        if self.target_mean is not None:
            feats = feats.float() - self.target_mean.to(feats.device)
        if self.normalize_targets:
            feats = F.layer_norm(feats.float(), (feats.shape[-1],))
        S, tok, D = feats.shape[1:]
        feats = feats.view(B, V, S, tok, D).permute(0, 2, 3, 1, 4).contiguous()  # [B, S, tok, V, D]
        return feats.view(B, S * tok, V * D)

    def _vjepa2_pixels(self, clips_tchw: list) -> torch.Tensor:
        """clips: list of arrays [T, C, H, W] uint8 -> pixel_values_videos [N, T, C, H, W]."""
        try:
            out = self.processor(videos=clips_tchw, return_tensors="pt")["pixel_values_videos"]
            if out.shape[0] != len(clips_tchw):
                raise ValueError("processor did not batch")
        except Exception:
            out = torch.cat([self.processor(videos=c, return_tensors="pt")["pixel_values_videos"] for c in clips_tchw], 0)
        return out.to(self.device)

    def _encode_vjepa2_clip(self, clips: np.ndarray) -> torch.Tensor:
        N, T = clips.shape[:2]
        tchw = [np.ascontiguousarray(c.transpose(0, 3, 1, 2)) for c in clips]
        px = self._vjepa2_pixels(tchw)
        emb = self.model.get_vision_features(pixel_values_videos=px)  # [N, S*tok, D]
        S = T // self.tubelet_size
        return emb.view(N, S, -1, emb.shape[-1])

    def _encode_vjepa2_perframe(self, clips: np.ndarray) -> torch.Tensor:
        N, T = clips.shape[:2]
        frames = clips[:, :: self.state_stride]  # [N, S, H, W, C]
        S = frames.shape[1]
        # duplicate every frame into a 2-frame tubelet so the tubelet embedding sees one image
        pairs = np.repeat(frames.reshape(N * S, 1, *frames.shape[2:]), self.tubelet_size, axis=1)
        tchw = [np.ascontiguousarray(c.transpose(0, 3, 1, 2)) for c in pairs]
        px = self._vjepa2_pixels(tchw)
        emb = self.model.get_vision_features(pixel_values_videos=px)  # [N*S, tok, D]
        return emb.view(N, S, -1, emb.shape[-1])

    def _levjepa_pixels(self, clips: np.ndarray) -> torch.Tensor:
        """uint8 [N, T, H, W, C] -> float [N, C, T, img, img] ImageNet-normalised."""
        x = torch.from_numpy(np.ascontiguousarray(clips)).to(self.device)
        x = x.permute(0, 4, 1, 2, 3).float() / 255.0  # [N, C, T, H, W]
        N, C, T, H, W = x.shape
        if (H, W) != (self.img_size, self.img_size):
            x = F.interpolate(
                x.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            ).view(N, T, C, self.img_size, self.img_size).permute(0, 2, 1, 3, 4)
        return (x - self._mean.to(x.device)) / self._std.to(x.device)

    def _encode_levjepa(self, clips: np.ndarray) -> torch.Tensor:
        N, T = clips.shape[:2]
        px = self._levjepa_pixels(clips)
        out = self.model(pixel_values=px)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        hidden = hidden[:, 1:]  # drop CLS
        feats = hidden.view(N, T, self.tokens_per_state, hidden.shape[-1])
        if self.state_stride > 1:
            feats = feats[:, self.state_stride - 1 :: self.state_stride]
        return feats


def build_target_encoder(vj_cfg, model=None, processor=None) -> TargetEncoder:
    return TargetEncoder(vj_cfg, model=model, processor=processor)

"""
Latent-action bottleneck variants for VLA-JEPA.

The VLM emits K latent-action tokens z in R^{H} (H = VLM hidden size). Upstream feeds
them straight into the predictor's `action_encoder` Linear(H -> predictor_dim). This module
sits in between and lets an experiment constrain the channel:

    none       identity (upstream behaviour)
    layernorm  LayerNorm(H) without affine, i.e. only remove scale/offset freedom
    lowrank    Linear(H -> d) -> Linear(d -> H); information bottleneck by rank
    vib        variational IB: Linear(H -> 2d) -> sample -> Linear(d -> H) + beta * KL
    vq         vector quantisation with a learned codebook in R^d, commitment loss
    drop       z := 0 (control: the predictor gets no information from the VLM)
    center     z - mu, where mu is a stop-gradient EMA of the per-slot mean of z. Removes the
               shared component that dominates z in the released checkpoints (RESULTS.md,
               11 Sep 2026) so the per-sample residual is what reaches the predictor.
    center_ln  center, then LayerNorm(H) with learnable affine and a scalar gain.

The EMA mean is a buffer of shape [K, H] (per token slot) or [1, H] (`center_per_slot: false`).
It is updated only in training mode, from the detached batch mean, with momentum
`center_momentum`; the first training batch initialises it exactly. In eval mode the stored
mean is used unchanged. A scalar `gain` (learnable if `center_learnable_gain`) rescales the
centered output so its norm can be matched to what the pretrained `action_encoder` expects.

Every variant maps back to R^{H} so the predictor and its pretrained `action_encoder`
weights stay untouched. Auxiliary losses are returned already weighted so the trainer can
sum the dict; unweighted values go into the metrics dict.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("none", "layernorm", "lowrank", "vib", "vq", "drop", "center", "center_ln")


class LatentActionBottleneck(nn.Module):
    def __init__(
        self,
        dim: int,
        kind: str = "none",
        bottleneck_dim: int = 32,
        vib_beta: float = 1e-3,
        vq_codebook_size: int = 512,
        vq_beta: float = 0.25,
        noise_std: float = 0.0,
        pre_layernorm: bool = False,
        center_momentum: float = 0.99,
        center_per_slot: bool = True,
        center_num_tokens: int = 24,
        center_gain: float = 1.0,
        center_learnable_gain: bool = False,
    ) -> None:
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"unknown bottleneck kind {kind!r}; choose from {KINDS}")
        self.kind = kind
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.vib_beta = vib_beta
        self.vq_beta = vq_beta
        self.noise_std = noise_std
        self.pre_norm = nn.LayerNorm(dim, elementwise_affine=False) if pre_layernorm else None

        if kind == "layernorm":
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        elif kind == "lowrank":
            self.down = nn.Linear(dim, bottleneck_dim)
            self.up = nn.Linear(bottleneck_dim, dim)
        elif kind == "vib":
            self.down = nn.Linear(dim, 2 * bottleneck_dim)
            self.up = nn.Linear(bottleneck_dim, dim)
        elif kind == "vq":
            self.down = nn.Linear(dim, bottleneck_dim)
            self.codebook = nn.Embedding(vq_codebook_size, bottleneck_dim)
            nn.init.uniform_(self.codebook.weight, -1.0 / vq_codebook_size, 1.0 / vq_codebook_size)
            self.up = nn.Linear(bottleneck_dim, dim)
        elif kind in ("center", "center_ln"):
            self.center_momentum = float(center_momentum)
            self.center_per_slot = bool(center_per_slot)
            rows = int(center_num_tokens) if self.center_per_slot else 1
            self.register_buffer("ema_mean", torch.zeros(rows, dim))
            self.register_buffer("ema_initialized", torch.zeros((), dtype=torch.bool))
            gain = torch.tensor(float(center_gain))
            if center_learnable_gain:
                self.gain = nn.Parameter(gain)
            else:
                self.register_buffer("gain", gain)
            if kind == "center_ln":
                self.norm = nn.LayerNorm(dim, elementwise_affine=True)

    # ------------------------------------------------------------------ helpers
    def _center(self, x: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """x: [B, K, H] float. Subtract the stop-gradient EMA mean (per slot or shared)."""
        B, K, H = x.shape
        if self.ema_mean.shape[0] not in (1, K):
            # slot count changed (another encoder gives another number of transitions)
            self.ema_mean = torch.zeros(K if self.center_per_slot else 1, H, device=x.device, dtype=self.ema_mean.dtype)
            self.ema_initialized.zero_()
        if self.training:
            with torch.no_grad():
                if self.ema_mean.shape[0] == 1:
                    batch_mean = x.mean(dim=(0, 1)).unsqueeze(0)
                else:
                    batch_mean = x.mean(0)
                batch_mean = batch_mean.to(self.ema_mean.dtype)
                if bool(self.ema_initialized):
                    self.ema_mean.mul_(self.center_momentum).add_(batch_mean, alpha=1.0 - self.center_momentum)
                else:
                    self.ema_mean.copy_(batch_mean)
                    self.ema_initialized.fill_(True)
        mu = self.ema_mean.to(x.dtype).unsqueeze(0)  # [1, K or 1, H], no grad
        out = x - mu
        metrics["metric/center_mean_norm"] = mu.norm(dim=-1).mean().detach()
        metrics["metric/center_residual_norm"] = out.norm(dim=-1).mean().detach()
        return out

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # accept an EMA buffer saved with another slot count (per-slot vs shared, or another K)
        key = prefix + "ema_mean"
        if key in state_dict and hasattr(self, "ema_mean") and tuple(state_dict[key].shape) != tuple(self.ema_mean.shape):
            self.ema_mean = torch.zeros_like(state_dict[key], device=self.ema_mean.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    def _vq(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h: [..., d] -> quantised (straight-through), loss, perplexity."""
        flat = h.reshape(-1, h.shape[-1])
        cb = self.codebook.weight  # [K, d]
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.t() + cb.pow(2).sum(1)[None, :]
        idx = dist.argmin(dim=1)
        q = self.codebook(idx).view_as(h)
        codebook_loss = F.mse_loss(q, h.detach())
        commit_loss = F.mse_loss(h, q.detach())
        loss = codebook_loss + self.vq_beta * commit_loss
        q_st = h + (q - h).detach()
        with torch.no_grad():
            counts = torch.bincount(idx, minlength=cb.shape[0]).float()
            p = counts / counts.sum().clamp(min=1)
            perplexity = torch.exp(-(p * torch.log(p.clamp(min=1e-10))).sum())
        return q_st, loss, perplexity

    # ------------------------------------------------------------------ forward
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        z: [B, K, H] latent-action tokens from the VLM.
        Returns (z_out [B, K, H], weighted_losses, metrics).
        """
        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}
        x = z.float() if self.kind not in ("none", "drop") else z
        if self.pre_norm is not None and self.kind not in ("none", "drop"):
            x = self.pre_norm(x)

        if self.kind == "none":
            out = z
        elif self.kind == "drop":
            out = torch.zeros_like(z)
        elif self.kind == "layernorm":
            out = self.norm(x)
        elif self.kind == "lowrank":
            out = self.up(self.down(x))
        elif self.kind == "vib":
            mu, logvar = self.down(x).chunk(2, dim=-1)
            logvar = logvar.clamp(-10.0, 10.0)
            if self.training:
                h = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            else:
                h = mu
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1).mean()
            losses["vib_kl_loss"] = self.vib_beta * kl
            metrics["metric/vib_kl_raw"] = kl.detach()
            out = self.up(h)
        elif self.kind == "vq":
            h = self.down(x)
            q, vq_loss, ppl = self._vq(h)
            losses["vq_loss"] = vq_loss
            metrics["metric/vq_perplexity"] = ppl
            out = self.up(q)
        elif self.kind == "center":
            out = self._center(x, metrics) * self.gain.to(x.dtype)
        elif self.kind == "center_ln":
            out = self.norm(self._center(x, metrics)) * self.gain.to(x.dtype)
        else:  # pragma: no cover
            raise RuntimeError(self.kind)

        if self.noise_std > 0 and self.training and self.kind != "drop":
            out = out + torch.randn_like(out) * self.noise_std

        return out.to(z.dtype), losses, metrics

    def extra_repr(self) -> str:
        return f"kind={self.kind}, dim={self.dim}, bottleneck_dim={self.bottleneck_dim}, noise_std={self.noise_std}"


def build_latent_bottleneck(cfg, dim: int) -> LatentActionBottleneck:
    """cfg: the `framework.latent_action` config node (may be None / missing)."""
    if cfg is None:
        return LatentActionBottleneck(dim=dim, kind="none")
    get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
    return LatentActionBottleneck(
        dim=dim,
        kind=get("bottleneck", "none"),
        bottleneck_dim=int(get("bottleneck_dim", 32)),
        vib_beta=float(get("vib_beta", 1e-3)),
        vq_codebook_size=int(get("vq_codebook_size", 512)),
        vq_beta=float(get("vq_beta", 0.25)),
        noise_std=float(get("noise_std", 0.0)),
        pre_layernorm=bool(get("pre_layernorm", False)),
        center_momentum=float(get("center_momentum", 0.99)),
        center_per_slot=bool(get("center_per_slot", True)),
        center_num_tokens=int(get("center_num_tokens", 24)),
        center_gain=float(get("center_gain", 1.0)),
        center_learnable_gain=bool(get("center_learnable_gain", False)),
    )

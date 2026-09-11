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

Every variant maps back to R^{H} so the predictor and its pretrained `action_encoder`
weights stay untouched. Auxiliary losses are returned already weighted so the trainer can
sum the dict; unweighted values go into the metrics dict.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("none", "layernorm", "lowrank", "vib", "vq", "drop")


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

    # ------------------------------------------------------------------ helpers
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
    )

"""
Anchor-Align for VLA-JEPA.

Port of the two losses from Dalal et al., "Generalizable VLA Finetuning via Representation
Anchoring and Language-Action Alignment" (arXiv:2607.13429, CC BY 4.0). Logic follows the
authors' starVLA implementation, MIT licensed:
  https://github.com/dwipddalal/Anchor-Align
  real_world_training/starVLA/model/framework/VLM4A/QwenGR00T.py
  (their snapshot derives from starVLA commit 5ce9e2ed59dc94de926c335d36a3681c243a58f7)

Differences from the source, all forced by VLA-JEPA's token layout:
  * the anchor mask excludes the 24 `<|action_i|>` latent-action positions and the 32
    `<|embodied_action|>` positions, not just padding: anchoring them would pin the latent
    action to a frozen VLM's output for a random new embedding;
  * the align position is the token right before the first latent-action token, not the
    last non-pad token (the sequence ends with the embodied tokens + generation prompt);
  * direction labels are computed from the normalised xyz deltas already in the batch
    (LIBERO/Droid actions are EEF deltas), with an optional per-axis zero point, instead
    of an external FK cache.
"""
import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DIRECTION_WORDS = ("forward", "backward", "left", "right", "up", "down")
IGNORE_INDEX = -100


# --------------------------------------------------------------------------- pure functions
def build_anchor_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    excluded_token_ids: Sequence[int],
) -> torch.Tensor:
    """Bool mask [B, L]: True on positions that should be anchored to the teacher
    (attended, and not one of the special latent/embodied action tokens)."""
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        keep &= attention_mask.bool()
    if len(excluded_token_ids) > 0:
        excl = torch.tensor(list(excluded_token_ids), device=input_ids.device, dtype=input_ids.dtype)
        keep &= ~torch.isin(input_ids, excl)
    return keep


def masked_layerwise_mse(
    student: Sequence[torch.Tensor],
    teacher: Sequence[torch.Tensor],
    keep_mask: torch.Tensor,
    sigma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean over layers of 0.5/sigma^2 * MSE(student, teacher) on kept positions.
    Returns (loss, per_layer_mse [num_layers]) in fp32."""
    assert len(student) == len(teacher), f"layer count mismatch: {len(student)} vs {len(teacher)}"
    m = keep_mask.unsqueeze(-1).float()
    denom = m.sum() * student[0].shape[-1]
    per_layer = []
    for s, t in zip(student, teacher):
        diff = (s.float() - t.float()).pow(2) * m
        per_layer.append(diff.sum() / denom.clamp(min=1.0))
    per_layer = torch.stack(per_layer)
    loss = (0.5 / (sigma**2)) * per_layer.mean()
    return loss, per_layer.detach()


def direction_labels(
    actions: torch.Tensor,
    min_norm: float = 0.0,
    axis_map: Sequence[int] = (0, 1, 2),
    zero_point: Optional[Sequence[float]] = None,
    flip_sign: Sequence[bool] = (False, False, False),
) -> torch.Tensor:
    """
    actions: [B, K, D] action chunk (any units). Uses dims `axis_map` = (x, y, z).
    Chunk-average the xyz delta, subtract `zero_point`, drop samples with L2 norm < min_norm,
    then dominant axis + sign -> class index into DIRECTION_WORDS:
        x>0 forward(0) x<0 backward(1) | y>0 left(2) y<0 right(3) | z>0 up(4) z<0 down(5)
    Returns long tensor [B] with IGNORE_INDEX for filtered samples.
    """
    xyz = actions[..., list(axis_map)].float().mean(dim=1)  # [B, 3]
    if zero_point is not None:
        xyz = xyz - torch.tensor(list(zero_point), device=xyz.device, dtype=xyz.dtype)
    sign_fix = torch.tensor([-1.0 if f else 1.0 for f in flip_sign], device=xyz.device, dtype=xyz.dtype)
    xyz = xyz * sign_fix
    norm = xyz.norm(dim=1)
    axis = xyz.abs().argmax(dim=1)  # 0,1,2
    val = torch.gather(xyz, 1, axis.unsqueeze(1)).squeeze(1)
    labels = axis * 2 + (val < 0).long()
    labels = torch.where(norm >= min_norm, labels, torch.full_like(labels, IGNORE_INDEX))
    return labels


def pre_action_positions(input_ids: torch.Tensor, action_token_ids: Sequence[int]) -> torch.Tensor:
    """Index [B] of the token immediately before the first latent-action token per row.
    Falls back to the last position if a row has no action token."""
    isin = torch.isin(input_ids, torch.tensor(list(action_token_ids), device=input_ids.device, dtype=input_ids.dtype))
    B, L = input_ids.shape
    first = torch.where(isin.any(dim=1), isin.float().argmax(dim=1), torch.full((B,), L, device=input_ids.device))
    return (first - 1).clamp(min=0, max=L - 1)


# --------------------------------------------------------------------------- module
class AnchorAlign(nn.Module):
    """
    Holds the frozen anchor teacher and the align projection.

    Config node `framework.anchor_align`:
        enable_anchor: bool        anchor_weight: 0.1   anchor_sigma: 1.0   anchor_layers: all|last
        anchor_on_video_batch: true
        teacher_source: base_vlm | pretrained_checkpoint   (see `sync_teacher_from`)
        enable_align: bool         align_weight: 0.02   align_min_xyz_norm: 0.0
        align_axis_map: [0,1,2]    align_zero_point: null | [x,y,z]   align_flip_sign: [f,f,f]
    """

    def __init__(self, cfg, vlm_interface: nn.Module, hidden_size: int):
        super().__init__()
        get = cfg.get if hasattr(cfg, "get") else (lambda k, d=None: getattr(cfg, k, d))
        self.enable_anchor = bool(get("enable_anchor", False))
        self.anchor_weight = float(get("anchor_weight", 0.1))
        self.anchor_sigma = float(get("anchor_sigma", 1.0))
        self.anchor_layers = str(get("anchor_layers", "all"))
        self.anchor_on_video_batch = bool(get("anchor_on_video_batch", True))
        self.teacher_source = str(get("teacher_source", "base_vlm"))

        self.enable_align = bool(get("enable_align", False))
        self.align_weight = float(get("align_weight", 0.02))
        self.align_min_xyz_norm = float(get("align_min_xyz_norm", 0.0))
        self.align_axis_map = tuple(int(i) for i in (get("align_axis_map", [0, 1, 2]) or [0, 1, 2]))
        zp = get("align_zero_point", None)
        self.align_zero_point = None if zp is None else tuple(float(v) for v in zp)
        self.align_flip_sign = tuple(bool(v) for v in (get("align_flip_sign", [False, False, False]) or [False] * 3))

        self.anchor_teacher: Optional[nn.Module] = None
        if self.enable_anchor:
            self.anchor_teacher = copy.deepcopy(vlm_interface)
            for p in self.anchor_teacher.parameters():
                p.requires_grad_(False)
            self.anchor_teacher.eval()
            # only hidden states are needed; drop the 300M-parameter lm_head (paper App. B.3)
            try:
                self.anchor_teacher.model.lm_head = nn.Identity()
            except Exception:
                pass

        self.direction_token_ids: List[int] = []
        if self.enable_align:
            self.align_dir_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            tok = vlm_interface.processor.tokenizer
            for w in DIRECTION_WORDS:
                ids = tok(w, add_special_tokens=False)["input_ids"]
                if len(ids) == 0:
                    raise RuntimeError(f"tokenizer returned no ids for direction word {w!r}")
                self.direction_token_ids.append(int(ids[0]))
            if len(set(self.direction_token_ids)) != len(DIRECTION_WORDS):
                raise RuntimeError(f"direction words are not distinct single tokens: {self.direction_token_ids}")

    # ----------------------------------------------------------------- lifecycle
    def train(self, mode: bool = True):
        super().train(mode)
        if self.anchor_teacher is not None:
            self.anchor_teacher.eval()
        return self

    @torch.no_grad()
    def sync_teacher_from(self, vlm_interface: nn.Module) -> None:
        """Copy the (possibly checkpoint-loaded) student VLM weights into the teacher.
        Call after loading a pretrained checkpoint when teacher_source == 'pretrained_checkpoint'."""
        if self.anchor_teacher is None:
            return
        src = {k: v for k, v in vlm_interface.state_dict().items() if not k.startswith("model.lm_head")}
        missing, unexpected = self.anchor_teacher.load_state_dict(src, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("model.lm_head")]
        if unexpected:
            raise RuntimeError(f"teacher sync: unexpected keys {unexpected[:5]}")

    def milestone_exclude_prefixes(self) -> List[str]:
        return ["anchor_align.anchor_teacher."]

    # ----------------------------------------------------------------- anchor
    @torch.no_grad()
    def teacher_hidden_states(self, qwen_inputs: Dict) -> List[torch.Tensor]:
        out = self.anchor_teacher(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[1:]  # skip the embedding layer
        if self.anchor_layers == "last":
            hs = hs[-1:]
        return [h.detach() for h in hs]

    def anchor_loss(
        self, student_hidden_states: Sequence[torch.Tensor], teacher_hidden_states: Sequence[torch.Tensor], keep_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s = list(student_hidden_states[1:])
        if self.anchor_layers == "last":
            s = s[-1:]
        loss, per_layer = masked_layerwise_mse(s, teacher_hidden_states, keep_mask, self.anchor_sigma)
        metrics = {"metric/anchor_raw": loss.detach(), "metric/anchor_last_layer_mse": per_layer[-1]}
        return self.anchor_weight * loss, metrics

    # ----------------------------------------------------------------- align
    def align_loss(
        self,
        last_hidden: torch.Tensor,
        pre_action_pos: torch.Tensor,
        actions: torch.Tensor,
        lm_head: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = last_hidden.shape[0]
        h = last_hidden[torch.arange(B, device=last_hidden.device), pre_action_pos]  # [B, H]
        cls = direction_labels(
            actions,
            min_norm=self.align_min_xyz_norm,
            axis_map=self.align_axis_map,
            zero_point=self.align_zero_point,
            flip_sign=self.align_flip_sign,
        )
        tok_ids = torch.tensor(self.direction_token_ids, device=last_hidden.device)
        targets = torch.where(cls >= 0, tok_ids[cls.clamp(min=0)], torch.full_like(cls, IGNORE_INDEX))
        proj = self.align_dir_proj(h.float())
        w = lm_head.weight.detach().to(proj.dtype)
        b = lm_head.bias.detach().to(proj.dtype) if getattr(lm_head, "bias", None) is not None else None
        logits = F.linear(proj, w, b)  # [B, vocab]
        loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
        with torch.no_grad():
            valid = cls >= 0
            if valid.any():
                pred = logits[:, tok_ids].argmax(dim=1)
                acc = (pred[valid] == cls[valid]).float().mean()
            else:
                acc = torch.zeros((), device=last_hidden.device)
            frac = valid.float().mean()
        if not torch.isfinite(loss):
            loss = torch.zeros((), device=last_hidden.device, dtype=torch.float32)
        metrics = {"metric/align_raw": loss.detach(), "metric/align_acc": acc, "metric/align_valid_frac": frac}
        return self.align_weight * loss, metrics

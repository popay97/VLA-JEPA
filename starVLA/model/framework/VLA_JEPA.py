# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
# Research extensions (target-encoder abstraction, latent bottleneck, Anchor-Align,
# configurable loss weights, diagnostics hooks) added on the `research` branch, 2026.
"""
VLA-JEPA framework: Qwen-VL + frozen video target encoder + action-conditioned latent
predictor (V-JEPA2-AC style) + GR00T flow-matching action head.

Latent-action path (per sample):
    VLM last hidden state at the K = (S-1) * num_action_tokens_per_timestep `<|action_i|>`
    positions  ->  latent bottleneck (optional)  ->  predictor `action_encoder`
    predictor input [z_k | s_k] for k < S-1, outputs at s_k positions predict s_{k+1}.

Config additions (all optional, defaults reproduce upstream behaviour):
    framework.vj2_model.encoder_type        vjepa2_clip | vjepa2_perframe | levjepa
    framework.vj2_model.normalize_targets   layer-norm encoder states (V-JEPA2-AC)
    framework.vj2_model.state_stride        subsample states (perframe / levjepa)
    framework.vj2_model.wm_loss_weight      default 0.1 (upstream hard-coded)
    framework.vj2_model.video_wm_loss_weight default 1.0 (video-only batches)
    framework.latent_action.*               see latent_bottleneck.py
    framework.anchor_align.*                see regularizers/anchor_align.py
Forward returns a dict; keys starting with "metric/" are logged but not summed into the loss.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.modules.world_model.target_encoders import build_target_encoder
from starVLA.model.modules.world_model.latent_bottleneck import build_latent_bottleneck
from starVLA.model.modules.regularizers.anchor_align import AnchorAlign, build_anchor_mask, pre_action_positions
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _get(node, key, default=None):
    if node is None:
        return default
    if hasattr(node, "get"):
        try:
            v = node.get(key, default)
            return default if v is None else v
        except Exception:
            pass
    return getattr(node, key, default)


@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - frozen video target encoder (V-JEPA 2 or LeVJEPA)
      - action-conditioned latent predictor (world model)
      - DiT flow-matching head for the action chunk
    """

    # modules that may legitimately be absent from an upstream checkpoint
    NEW_MODULE_PREFIXES = ["latent_bottleneck.", "anchor_align."]

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        fw = config.framework
        vj_cfg = fw.vj2_model

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = _get(vj_cfg, "embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=vj_cfg.special_action_token,
            max_action_tokens=fw.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token,
        )
        self._special_token_ids = list(self.action_token_ids) + [self.embodied_action_token_id]

        hidden = self.qwen_vl_interface.model.config.hidden_size
        fw.action_model.diffusion_model_cfg.cross_attention_dim = hidden
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = fw.action_model.future_action_window_size
        self.past_action_window_size = fw.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # ---- frozen target encoder -------------------------------------------------
        self.target_encoder = build_target_encoder(vj_cfg)
        self.vj_encoder = self.target_encoder.model  # registered submodule -> `vj_encoder.*` keys
        self.vj_processor = self.target_encoder.processor
        self.num_frames = int(vj_cfg.num_frames)
        self.num_states = self.target_encoder.num_states(self.num_frames)
        self.num_transitions = self.num_states - 1
        if self.num_transitions < 1:
            raise ValueError(f"need >= 2 latent states, got {self.num_states} from {self.num_frames} frames")
        if len(action_tokens) < self.num_transitions:
            raise ValueError(f"only {len(action_tokens)} `<|action_i|>` tokens for {self.num_transitions} transitions")

        # ---- predictor ---------------------------------------------------------------
        self.num_action_tokens_per_timestep = int(vj_cfg.num_action_tokens_per_timestep)
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.num_states,
            img_size=(self.target_encoder.img_size, self.target_encoder.img_size),
            patch_size=self.target_encoder.patch_size,
            tubelet_size=1,
            depth=vj_cfg.depth,
            num_heads=vj_cfg.num_heads,
            embed_dim=self.target_encoder.hidden_size * 2,  # two views concatenated
            action_embed_dim=hidden,
            num_add_tokens=self.num_action_tokens_per_timestep,
            use_activation_checkpointing=bool(_get(vj_cfg, "predictor_activation_checkpointing", False)),
        )
        self.replace_prompt = "".join(
            tok * self.num_action_tokens_per_timestep for tok in action_tokens[: self.num_transitions]
        )
        self.embodied_replace_prompt = embodied_action_token * int(vj_cfg.num_embodied_action_tokens_per_instruction)

        self.wm_loss_weight = float(_get(vj_cfg, "wm_loss_weight", 0.1))
        self.video_wm_loss_weight = float(_get(vj_cfg, "video_wm_loss_weight", 1.0))

        # ---- research modules --------------------------------------------------------
        self.latent_bottleneck = build_latent_bottleneck(_get(fw, "latent_action", None), dim=hidden)
        aa_cfg = _get(fw, "anchor_align", None)
        self.anchor_align: Optional[AnchorAlign] = None
        if aa_cfg is not None and (bool(_get(aa_cfg, "enable_anchor", False)) or bool(_get(aa_cfg, "enable_align", False))):
            self.anchor_align = AnchorAlign(aa_cfg, self.qwen_vl_interface, hidden)

        logger.info(
            f"[VLA_JEPA] encoder={self.target_encoder.encoder_type} states={self.num_states} "
            f"grid={self.target_encoder.grid} tokens/transition={self.num_action_tokens_per_timestep} "
            f"bottleneck={self.latent_bottleneck.kind} wm_w={self.wm_loss_weight} "
            f"anchor={'on' if self.anchor_align is not None and self.anchor_align.enable_anchor else 'off'} "
            f"align={'on' if self.anchor_align is not None and self.anchor_align.enable_align else 'off'}"
        )

    # ------------------------------------------------------------------ lifecycle hooks
    def train(self, mode: bool = True):
        super().train(mode)
        self.vj_encoder.eval()  # frozen target encoder never trains
        if self.anchor_align is not None:
            self.anchor_align.train(mode)
        return self

    def milestone_exclude_prefixes(self) -> List[str]:
        return self.anchor_align.milestone_exclude_prefixes() if self.anchor_align is not None else []

    def on_pretrained_loaded(self) -> None:
        """Called by the trainer after `load_pretrained_backbones`."""
        if self.anchor_align is not None and self.anchor_align.enable_anchor and self.anchor_align.teacher_source == "pretrained_checkpoint":
            self.anchor_align.sync_teacher_from(self.qwen_vl_interface)
            logger.info("[VLA_JEPA] anchor teacher synced from the pretrained checkpoint's VLM")

    # ------------------------------------------------------------------ tokenizer
    def expand_tokenizer(
        self,
        tokenizer: AutoTokenizer,
        special_action_token: str = "<|action_{}|>",
        max_action_tokens: int = 32,
        embodied_action_token: str = "<|embodied_action|>",
    ):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_ids.append(tokenizer.convert_tokens_to_ids(action_token_i))

        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    # ------------------------------------------------------------------ shared pieces
    def _build_inputs(self, batch_images, instructions, has_actions: bool):
        if has_actions:
            return self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images,
                instructions=instructions,
                prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
            )
        return self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={"{actions}": self.replace_prompt},
            prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""),
        )

    def _split_states(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = states.shape[1] // self.num_states
        return states[:, : tok * self.num_transitions], states[:, tok:]

    def encode_states(self, batch_videos: np.ndarray, device) -> torch.Tensor:
        """[B, V, T, H, W, 3] uint8 -> [B, S*tok, V*D] on `device` (no grad)."""
        with torch.no_grad():
            states = self.target_encoder.encode(batch_videos)
        return states.to(device)

    def world_model_loss(self, z: torch.Tensor, input_states: torch.Tensor, gt_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.vj_predictor(input_states, z)
        return F.l1_loss(pred.float(), gt_states.float(), reduction="mean")

    # ------------------------------------------------------------------ training forward
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_images = [ex["image"] for ex in examples]  # [B, [PIL.Image]]
        batch_videos = np.stack([ex["video"] for ex in examples])  # [B, V, T, H, W, 3]
        instructions = [ex["lang"] for ex in examples]
        has_actions = "action" in examples[0]
        actions = [ex["action"] for ex in examples] if has_actions else None
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        qwen_inputs = self._build_inputs(batch_images, instructions, has_actions)
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs.get("attention_mask", None)
        action_mask = torch.isin(input_ids, torch.tensor(self.action_token_ids, device=input_ids.device))
        embodied_mask = input_ids == self.embodied_action_token_id

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            hidden_states = qwenvl_outputs.hidden_states
            last_hidden = hidden_states[-1]  # [B, L, H]
            B, _, H = last_hidden.shape
            z = last_hidden[action_mask].view(B, -1, H)  # [B, (S-1)*k, H]
            embodied_action_tokens = last_hidden[embodied_mask].view(B, -1, H)

        losses: Dict[str, torch.Tensor] = {}
        metrics: Dict[str, torch.Tensor] = {}

        # latent bottleneck
        z, bn_losses, bn_metrics = self.latent_bottleneck(z)
        losses.update(bn_losses)
        metrics.update(bn_metrics)

        # world model
        states = self.encode_states(batch_videos, last_hidden.device)
        input_states, gt_states = self._split_states(states)
        wm_loss = self.world_model_loss(z, input_states, gt_states)
        losses["wm_loss"] = wm_loss * (self.wm_loss_weight if has_actions else self.video_wm_loss_weight)

        # anchor
        if self.anchor_align is not None and self.anchor_align.enable_anchor and (has_actions or self.anchor_align.anchor_on_video_batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_hs = self.anchor_align.teacher_hidden_states(qwen_inputs)
            keep = build_anchor_mask(input_ids, attention_mask, self._special_token_ids)
            a_loss, a_metrics = self.anchor_align.anchor_loss(hidden_states, teacher_hs, keep)
            losses["anchor_loss"] = a_loss
            metrics.update(a_metrics)

        if not has_actions:
            return {**losses, **metrics}

        # action head
        with torch.autocast("cuda", dtype=torch.float32):
            actions_t = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions_t[:, -(self.future_action_window_size + 1) :, :]
            repeated_diffusion_steps = _get(self.config.trainer, "repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state_t.repeat(repeated_diffusion_steps, 1, 1)
            action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated)
        losses["action_loss"] = action_loss

        # align
        if self.anchor_align is not None and self.anchor_align.enable_align:
            pos = pre_action_positions(input_ids, self.action_token_ids)
            al_loss, al_metrics = self.anchor_align.align_loss(
                last_hidden, pos, actions_target, self.qwen_vl_interface.model.lm_head
            )
            losses["align_loss"] = al_loss
            metrics.update(al_metrics)

        return {**losses, **metrics}

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def world_model_terms(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        """Everything the diagnostics need from one batch, without gradients."""
        batch_images = [ex["image"] for ex in examples]
        batch_videos = np.stack([ex["video"] for ex in examples])
        instructions = [ex["lang"] for ex in examples]
        has_actions = "action" in examples[0]
        qwen_inputs = self._build_inputs(batch_images, instructions, has_actions)
        input_ids = qwen_inputs["input_ids"]
        action_mask = torch.isin(input_ids, torch.tensor(self.action_token_ids, device=input_ids.device))
        embodied_mask = input_ids == self.embodied_action_token_id
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
            last_hidden = out.hidden_states[-1]
            B, _, H = last_hidden.shape
            z_raw = last_hidden[action_mask].view(B, -1, H)
            embodied = last_hidden[embodied_mask].view(B, -1, H)
        z, _, _ = self.latent_bottleneck(z_raw)
        states = self.encode_states(batch_videos, last_hidden.device)
        input_states, gt_states = self._split_states(states)
        terms = dict(
            z=z,
            z_raw=z_raw,
            embodied=embodied,
            input_states=input_states,
            gt_states=gt_states,
            hidden_states=out.hidden_states,
            input_ids=input_ids,
            attention_mask=qwen_inputs.get("attention_mask", None),
            pre_action_pos=pre_action_positions(input_ids, self.action_token_ids),
        )
        if has_actions:
            actions_t = torch.tensor(np.array([ex["action"] for ex in examples]), device=last_hidden.device, dtype=torch.float32)
            terms["actions_target"] = actions_t[:, -(self.future_action_window_size + 1) :, :]
        return terms

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> dict:
        """Single forward through the VLM, then flow-matching sampling of the action chunk."""
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
        )
        embodied_mask = qwen_inputs["input_ids"] == self.embodied_action_token_id

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_mask].view(B, -1, H)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {
            "normalized_actions": normalized_actions,
            "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy(),
        }

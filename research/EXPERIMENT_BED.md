# Experiment bed: characterizing the latent-action bottleneck in VLA-JEPA

Budget: 5000 A100-hours (EuroHPC). Fork: https://github.com/popay97/VLA-JEPA, branch
`research`. Upstream: https://github.com/ginwind/VLA-JEPA (starVLA-based, no LICENSE file in
the repo; README badge says Apache-2.0, pyproject says MIT).

## Known defects to fix before any run (Phase 0)

| Issue | Where | Fix |
|---|---|---|
| Scheduler stepped twice per iteration, so `cosine_with_min_lr` bottoms out at half of `max_train_steps` and climbs back | `starVLA/training/train_jevla_cotrain.py` `_train_step` | one `lr_scheduler.step()` per iteration, or `num_training_steps = 2 * max_train_steps` |
| World-model weight hard-coded (`* 0.1`), `loss_scale` YAML dead | `VLA_JEPA.py:243` | read from config |
| SSv2 captions from `test-answers.csv`, most videos get a placeholder caption | `scripts/configs/vlajepa_pretrain.yaml`, `video_datasets.py` | use train labels |
| Target encoder is fully bidirectional over the 8-frame clip: predictor inputs and L1 targets leak future frames | `VLA_JEPA.py` `get_vision_features` on the whole clip | see encoder axis below |
| Paper says uniform flow-matching timesteps; code samples Beta(1.5, 1) | `GR00T_ActionHeader.py` | document, keep |

## Experiment axes

1. **Target encoder** (leakage): V-JEPA 2 clip-level (current, leaky) / V-JEPA 2 per-frame
   (Meta's `forward_target` duplication trick, layer-normed targets) / LeVJEPA block-causal
   ViT-L (`galilai-group/LeVJEPA-VideoMix-Large`, tubelet 1, 224 px, 14x14 grid, ImageNet
   norm, CC BY-NC 4.0).
2. **Bottleneck on z**: none (current `Linear(2048,1024)`) / LayerNorm / low-rank projection
   d in {8, 32, 128} / KL (VIB) / VQ / no-z control (predictor without action tokens).
3. **Tokens per transition**: 1 / 3 / 8 (current 8; with tubelet 1 there are 7 transitions
   matching the 7-action chunk).
4. **Future prediction**: teacher-forced next state (current) / strided future chunk
   (Zero-WAM IFP style) / autoregressive rollout loss (V-JEPA2-AC `auto_steps`).
5. **Representation preservation**: none / Anchor / Anchor + Align
   (see `ANCHOR_ALIGN_INTEGRATION.md`).

## Diagnostics (run on every checkpoint)

- z-dependence: replace z with noise, shuffle z across the batch, drop z; measure WM loss
  and action MAE (LAWA-style).
- Scene-cut test: predictor on clips with a hard cut; a working z should not predict
  across the cut (Garrido et al.).
- CCA / linear probe from z to ground-truth action chunk (LAWM-style CCA ~0.9 target).
- Text-token CKA per layer against the frozen VLM, linear-probe R^2 for actions per
  layer, direction-word probe (Anchor-Align App. D).
- GQA accuracy of the VLM head (catastrophic-forgetting curve).

## Budget plan (A100-hours, ~2x overhead on compute estimates)

| Phase | Content | Cost |
|---|---|---|
| 0 | smoke test, fixes, diagnostics, align threshold statistics | ~50 |
| 1 | LIBERO-only fine-tune sweep: encoder x bottleneck x preservation, LIBERO-Plus eval | ~800 |
| 2 | future-prediction variants and tokens per transition on the best Phase 1 configs | ~600 |
| 3 | two best configs, full Droid + SSv2 pretraining with fixed captions | ~1500 |
| margin | reruns, seeds | ~2000 |

Per-run reference costs (35% MFU, 2x overhead): pretrain B=8 50k steps ~180; LIBERO ft 120k
~210, 30k ~55; SimplerEnv 30k B=32 ~210; LIBERO eval ~10, LIBERO-Plus ~40-60.

## Infrastructure prerequisites (Leonardo)

A separate engineering brief, "VLA-JEPA on Leonardo" (Claude artifact
https://claude.ai/code/artifact/8533f648-d965-4c0d-ad61-70ee6c141d3a, reviewed 8-9 Sep 2026,
shared within the org), covers the facility side. Its conclusions that gate this bed:

- Resume is broken as shipped: `_save_checkpoint` writes weights only
  (`train_jevla_cotrain.py:209-217`), `_load_checkpoint` calls `accelerator.load_state` on a
  directory never written. Any multi-day run needs `save_state` + dataloader state +
  `completed_steps`, a USR1 trap and `--requeue`. This is Phase 0 step one.
- Leonardo Booster A100s are 64 GB, not 80 GB. Budget the frozen Anchor teacher (~4.4 GB
  bf16 plus activations) against per-device batch 8 with gradient checkpointing.
- Re-encode DROID to the consumed resolution (2 views, 256 px, short GOP, LeRobot v2.1)
  before transfer: 1.7 TB RLDS -> 200-400 GB. Stage to `$SCRATCH`; no network from GPU nodes.
- Run shape: `boost_qos_lprod`, 8 nodes x 4 GPUs, per-device 8, global 256, 4-day chunks.
- One correction to the brief: it warns about an EMA-updated JEPA target encoder. In
  VLA-JEPA the V-JEPA 2 encoder is a plain `AutoModel.from_pretrained` called under
  `torch.no_grad()` (`VLA_JEPA.py:81,191-192`), no EMA, so nothing extra to checkpoint
  there. It is also never set to `.eval()` or `requires_grad_(False)`; harmless today, but
  do both explicitly when touching the encoder axis.

## Third-party code policy

Port small pieces with attribution (see `THIRD_PARTY_NOTICES.md`). No submodules: the only
external code we need is ~150 lines from Anchor-Align (MIT) and HF `trust_remote_code`
checkpoints for LeVJEPA / V-JEPA 2.1, which are dependencies, not vendored code.

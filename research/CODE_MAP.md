# Code map for the `research` branch

What changed relative to upstream `ginwind/VLA-JEPA`, where it lives, and how to drive it.
Defaults reproduce the upstream LIBERO recipe exactly; every switch below is opt-in.

## Framework (`starVLA/model/framework/VLA_JEPA.py`, rewritten)

| Piece | File | Config | Notes |
|---|---|---|---|
| Target encoder abstraction | `starVLA/model/modules/world_model/target_encoders.py` | `framework.vj2_model.encoder_type` = `vjepa2_clip` (upstream) / `vjepa2_perframe` (Meta V-JEPA2-AC per-frame trick, leak-free) / `levjepa` (block-causal LeVJEPA, 224 px, 14x14); `normalize_targets`, `state_stride` | The HF model is still registered as `vj_encoder` so upstream checkpoints load. `num_states = T//tubelet` (clip) or `T//state_stride`. |
| Latent bottleneck on z | `starVLA/model/modules/world_model/latent_bottleneck.py` | `framework.latent_action.bottleneck` = `none`/`layernorm`/`lowrank`/`vib`/`vq`/`drop`, `bottleneck_dim`, `vib_beta`, `vq_*`, `noise_std`, `pre_layernorm` | Sits between the VLM's `<|action_i|>` hidden states and the predictor's `action_encoder`; maps back to H so pretrained predictor weights are untouched. Aux losses returned already weighted; raw values as `metric/*`. |
| Anchor-Align | `starVLA/model/modules/regularizers/anchor_align.py` | `framework.anchor_align.*` (`enable_anchor`, `anchor_weight`, `anchor_layers`, `teacher_source`, `enable_align`, `align_weight`, `align_min_xyz_norm`, `align_zero_point`, ...) | Frozen deepcopy teacher (lm_head removed) under `anchor_align.anchor_teacher.`; excluded from weights-only exports; `teacher_source=pretrained_checkpoint` syncs the teacher after the pretrained load via `on_pretrained_loaded()`. Align position = token before the first `<|action_0|>`. See `ANCHOR_ALIGN_INTEGRATION.md`. |
| Loss weights | `VLA_JEPA.py` | `framework.vj2_model.wm_loss_weight` (0.1), `video_wm_loss_weight` (1.0) | Upstream hard-coded these. |
| Output contract | `VLA_JEPA.forward` | | Returns a dict; the trainers sum every key except `metric/*` (`trainer_tools.sum_losses`). |
| Diagnostics hook | `VLA_JEPA.world_model_terms(examples)` | | No-grad access to z, z_raw, states, hidden states, positions, action targets for `research/diagnose.py`. |
| `from_pretrained` | `base_framework.py` | | Forces `anchor_align.enable_anchor=False` (teacher is not in exported weights). |

`framework.vj2_model.num_action_tokens_per_timestep` (8 upstream) and `num_frames` (8) keep their
meaning; `replace_prompt` is built from `num_states - 1` transitions, so per-frame arms use 7
distinct `<|action_i|>` tokens automatically. The predictor has no geometry-dependent learned
tensors (sincos pos-embed, mask rebuilt), so pretrained predictor weights load for every arm;
`trainer.skip_load_modules: vj_predictor` re-initialises it on purpose.

## Trainers (`starVLA/training/train_starvla.py`, `train_jevla_cotrain.py`)

* `ResumableCheckpointing` mixin (`starVLA/training/trainer_utils/checkpointing.py`): full
  training state via `accelerator.save_state` under `<run>/checkpoints/state/step_N/` with
  `meta.json` + `latest` pointer, rotation (`trainer.keep_last_states`), auto-resume
  (`trainer.auto_resume`, or `trainer.resume_from_checkpoint`), stateful dataloaders
  (`trainer.use_stateful_dataloader`, torchdata), save triggers: `save_interval` (also exports
  `steps_N_pytorch_model.pt`), `save_every_minutes`, `STOP_AND_SAVE` sentinel in the checkpoint
  dir, SIGUSR1/SIGTERM. `final_model/pytorch_model.pt` + `DONE` marker at the end. The LR
  scheduler is registered for checkpointing and stepped once per optimizer update.
* `train_jevla_cotrain.py`: the second `lr_scheduler.step()` per iteration was removed (it
  halved the schedule horizon).
* `trainer_tools.load_pretrained_backbones`: full loads are `strict=False` but only tolerate
  missing keys under `model.NEW_MODULE_PREFIXES` (`latent_bottleneck.`, `anchor_align.`) or
  `trainer.skip_load_modules`; any other missing/unexpected key raises.
* `build_param_lr_groups` skips `requires_grad=False` parameters (frozen encoder, teacher), so
  DeepSpeed does not allocate optimizer state for them. LR groups may name
  `latent_bottleneck` and `anchor_align`.
* The periodic action-MAE probe uses one fixed batch taken at start (upstream consumed a
  training batch on rank 0 only, desynchronising the ranks' dataloaders).
* `--extra_yaml a.yaml b.yaml` overlays merged onto `--config_yaml` before CLI dotlist overrides.
* `datasets.vla_data.num_workers` / `datasets.video_data.num_workers` (default 4).

## Configs

`scripts/configs/research/libero_ft_base.yaml` (Leonardo paths, 1 node x 4 GPU, 30k steps) +
one overlay per arm in `scripts/configs/research/arms/`: `base_30k`, `base_120k`, `noz`, `ln`,
`ln_targets`, `proj8/32/128`, `vib32`, `vq32`, `perframe`, `perframe_s2`, `levjepa`, `anchor`,
`anchor_align`, `anchor_ptteacher`, `tpt1`.

## Leonardo (`research/slurm/`)

`env.sh` (paths, offline caches, venv), `setup_env.sh` (cu121 venv, optional LIBERO venv),
`download_models.sh` (login node), `stage_data.sbatch` (LIBERO LeRobot to `$SCRATCH`),
`train.sbatch` (1 node, 24 h, `--requeue`, USR1 trap -> sentinel -> save -> requeue unless
`DONE`), `submit_arm.sh <arm>` (train + dependent diagnose + eval), `eval_libero.sbatch`,
`diagnose.sbatch`, `resume_test.sbatch` (T1 acceptance test on the dbg QoS).

## Diagnostics (`research/`)

`diagnose.py` (world-model controls z_zeros/noise/shuffle/tokshuffle/scene_cut/copy_last,
ridge R^2 + CCA z -> actions, direction probe, per-layer text CKA vs the base VLM, action MAE)
with the maths in `diag_utils.py`; `align_stats.py` (threshold / zero point for the align loss
from the parquet files); `collect_results.py` (summary table).

## Tests

`PYTHONPATH=. python -m pytest tests -q` (CPU only, fakes for VLM / encoder / action head):
bottleneck, target encoders, anchor-align, loader/optimizer groups, checkpoint save-resume
roundtrip, diagnostics maths, and a full `VLA_JEPA.forward` smoke test for every switch.

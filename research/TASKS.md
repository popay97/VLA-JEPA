# Task list: six weeks, 5,000 GPU-hours, ~9-day queue

Decision (10 Sep 2026): **no pretraining replication.** All experiments are LIBERO
fine-tunes from the released `ginwind/VLA-JEPA` `Pretrain` checkpoint, which is what the
paper's own LIBERO recipe starts from (`scripts/configs/vlajepa_libero_ft.yaml`,
`train_starvla.py`, 8 GPUs x batch 8, 120k steps, predictor lr 5e-4, world-model loss
active on LIBERO clips). DROID, SSv2 and the 8-node run are out of scope unless CINECA
grants an extension. The released `LIBERO` checkpoint is the diagnostic baseline and needs
no training at all.

Why this is enough for the research question: the world-model loss and the latent action
tokens are exercised during fine-tuning, so every intervention (bottleneck on z, encoder
swap, anchor, strided future) can be compared on identical data and compute. What we lose
is the ability to say anything about pretraining-scale effects; that is acceptable.

## Cost model (measure in T3, these are planning numbers)

| Job | GPUs | Wall | GPU-h |
|---|---|---|---|
| Short fine-tune, 30k steps, batch 8 x 4 | 4 | ~14 h | ~55 |
| Full fine-tune, 120k steps, batch 8 x 8 | 8 | ~26 h | ~210 |
| LIBERO eval (4 suites x 10 tasks x 50 trials) | 1 | ~10 h | ~10 |
| LIBERO-Plus eval | 1 | ~40-60 h | ~50 |
| Diagnostics per checkpoint | 1 | ~2 h | ~2 |

Budget: ~20 short runs (1,100) + 4 full runs (840) + 24 LIBERO evals (240) + 8 LIBERO-Plus
(400) + diagnostics (100) = ~2,700 GPU-h, leaving ~2,300 for seeds, reruns, and a second
wave. Monthly full-priority quota is ~2,460 GPU-h; wave 1 in September, wave 2 in October.

Queue rule: many 1-node jobs beat one 8-node job (same billing, more slots). Anything
over 24 h uses `boost_qos_lprod` or a `--dependency=afterany` chain with resume.

## T0. Today, no GPU needed (parallel)

- [ ] Email superc@cineca.it: project extension past 24 Oct, requeue permission, `$WORK`
      quota bump. Mention the 9-day queue estimate as the reason.
- [ ] On the Leonardo login node: `git clone` the fork to `$WORK/VLA-JEPA` (branch
      `research`); download to `$FAST/models/`: Qwen3-VL-2B-Instruct,
      `facebook/vjepa2-vitl-fpc64-256`, `galilai-group/LeVJEPA-VideoMix-Large`,
      `ginwind/VLA-JEPA` (Pretrain + LIBERO folders). Download LIBERO LeRobot datasets
      (IPEC-COMMUNITY libero collection) to `$SCRATCH/libero/`, add `modality.json` from
      `examples/LIBERO`. Long downloads in an `lrd_all_serial` job (login has a 600
      CPU-second limit).
- [ ] Build the runtime in `$FAST/venv` with `cu121` wheels (driver is CUDA 12.2):
      torch, transformers >= 4.57, accelerate, deepspeed, flash-attn (compile once, or
      reuse the `cineca-ai` build approach), qwen-vl-utils, av, decord. Build a second
      env or a Singularity image for LIBERO and LIBERO-Plus eval with EGL. Test both in a
      `boost_qos_dbg` job with `HF_HUB_OFFLINE=1`.

## T1. Code, this week (dev loop = `boost_qos_dbg`, 30 min, starts instantly)

Status 11 Sep: all T1 code is on `research` with 44 CPU tests passing (`research/CODE_MAP.md`).
What remains is cluster-side: run `resume_test.sbatch` once T0 staging is done, and set
`align_min_xyz_norm` / `align_zero_point` from `research/align_stats.py` before submitting
`anchor_align`.

- [x] (code, 11 Sep; cluster test pending via `research/slurm/resume_test.sbatch`) Resume: `accelerator.save_state`/`load_state`, `StatefulDataLoader`, persist
      `completed_steps`, save on `STOP_AND_SAVE` sentinel and every N minutes, `--requeue`
      + `--signal=B:USR1@900` in the sbatch template. Prove: 200 steps, kill at 100,
      resume, continuous LR and loss.
- [x] Scheduler: fix the double `lr_scheduler.step()` in `train_jevla_cotrain.py`
      (single-batch `train_starvla.py` is unaffected but keep both correct).
- [x] Config plumbing: `wm_loss` weight from YAML instead of the hard-coded 0.1;
      `framework.latent_action.*` switches for the bottleneck arms; `framework.anchor_align.*`
      per `ANCHOR_ALIGN_INTEGRATION.md`; `framework.vj2_model.encoder_type`
      in {`vjepa2_clip`, `vjepa2_perframe`, `levjepa`}.
- [x] Bottleneck module on z before `action_encoder`: identity | LayerNorm | low-rank
      `Linear(2048,d)->Linear(d,1024)` | VIB (KL) | VQ | drop-z (predictor without action
      tokens).
- [x] Encoder arms: per-frame V-JEPA 2 (duplicate each frame into a 2-frame tubelet,
      layer-norm targets, as Meta's `forward_target`); LeVJEPA wrapper (drop CLS, 14x14
      grid, ImageNet norm, `attn_mode="block_causal"`), predictor grid follows encoder.
- [x] Anchor + Align port (`starVLA/model/modules/regularizers/anchor_align.py`), mask
      out action and embodied token positions, teacher excluded from checkpoints.
- [x] Diagnostics script (`research/diagnose.py`): z noise / shuffle / drop on the WM loss
      and action MAE; scene-cut test; linear probe and CCA from z to the action chunk;
      per-layer text-token CKA to the frozen VLM; direction-word probe. Runs on any
      checkpoint on 1 GPU.
- [x] Sbatch templates under `research/slurm/`: train (1 node, 24 h, requeue chain),
      train-lprod (4 days), eval, diagnose. Log to `$WORK/runs/<name>/`.

## T2. Baseline before any training (submit as soon as T0 data is staged)

- [ ] Diagnostics on the released LIBERO checkpoint: this is the headline number for
      "does z carry anything today".
- [ ] LIBERO eval of the released LIBERO checkpoint with our eval env to validate the
      harness against the paper's reported success rates.
- [ ] Align threshold statistics: distribution of chunk-averaged xyz norm over LIBERO.

## T3. Wave 1, submit by ~12 Sep so it starts ~19-21 Sep (September quota)

All 30k-step fine-tunes from the Pretrain checkpoint, 1 node each, identical data and
seed. Names are config file names under `scripts/configs/research/`.

- [ ] `base_30k` (paper recipe, shortened) and `base_120k` (paper recipe exact) to
      calibrate whether 30k ranks arms the same way 120k does.
- [ ] `noz` (drop-z control), `ln`, `proj8`, `proj32`, `proj128`
- [ ] `perframe` (V-JEPA 2 per-frame, leak-free), `levjepa`
- [ ] `anchor` (lambda 0.1), `anchor_align`
- [ ] Every job ends by launching its own diagnostics job and LIBERO eval job via
      `--dependency=afterok`.

Twelve jobs x ~55 GPU-h + evals = ~800 GPU-h. Re-derive from the measured step time of
the first job to finish and adjust the wave-2 count.

## T4. Wave 2, submit ~1 Oct (October quota)

- [ ] Best bottleneck x best encoder, with and without anchor: 4 arms x 3 seeds at 30k.
- [ ] Two full 120k runs of the top two configs for the paper-comparable number.
- [ ] Strided future prediction and rollout-loss variants on the best config.
- [ ] LIBERO-Plus on the top 4 checkpoints and the released baseline.
- [ ] `tpt1`: tubelet 1 with 1 token per transition (7 x 1) versus 3 x 8, on the best encoder.

## T5. Continuous

- [ ] Keep at least two jobs pending at all times; `squeue --start` daily.
- [ ] Prune: keep final + best milestone per arm in `$WORK/milestones`, three rolling
      full-state saves in `$FAST/ckpt` per live job.
- [ ] Sync `$WORK/runs/*/tensorboard` and results to the HP box weekly.
- [ ] Write results into `research/RESULTS.md` as they land; update `WHY_THIS_FORK.md`
      when a conclusion changes.

## Out of scope unless an extension is granted

- Droid + SSv2 pretraining of the best config (Phase 3 in `EXPERIMENT_BED.md`).
- SimplerEnv and real-world evaluation.

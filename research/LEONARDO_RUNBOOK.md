# Leonardo runbook: training with 1 TB of quota and 2-day walltimes

Companion to the infrastructure brief linked in `EXPERIMENT_BED.md`. This file turns it
into a tactic and an ordered task list for our sweep. Facility figures should be
re-checked with `cindata` and `sinfo` before the first job.

## The tactic in six rules

1. **Space is a tiering problem, not a size problem.** Three areas, three roles.
   `$SCRATCH` (no quota, 40-day idle cleanup, no backup) holds datasets. `$FAST`
   (1 TB NVMe) holds everything hot and regenerable: venv or container, base models,
   rolling full-state checkpoints. `$WORK` (1 TB) holds what must survive: code at a tag,
   weights-only milestones, logs, eval results, dataset caches. `$HOME` (50 GB) holds
   nothing but dotfiles; move `HF_HOME`, `TORCH_HOME`, `TMPDIR` off it.
2. **Ship only what the phase needs.** Phases 0 to 2 are LIBERO-only fine-tunes from the
   released pretrain checkpoint. They need the LIBERO LeRobot data (tens of GB), two base
   models, the ~5 GB pretrain checkpoint and SSv2 if the world-model arm co-trains on
   video. DROID (1.7 TB as RLDS) is needed only in Phase 3, so the big transfer never gates
   the start.
3. **Shrink at the source.** Re-encode DROID to what the model consumes: two views, 256 px,
   short GOP (`libsvtav1 -g 2 -crf 30` or equivalent), LeRobot v2.1 layout. 1.7 TB becomes
   200 to 400 GB and decode cost drops about eightfold. Do this on Hetzner or the HP box,
   never on Booster.
4. **Checkpoints, not data, are what fills the quota in a sweep.** Full training state for
   2.6B params is about 31 GB (fp32 master weights plus AdamW moments). Weights-only bf16 is
   about 5.2 GB. Policy: at most three rolling full-state saves per live job in `$FAST`
   (93 GB), one weights-only milestone per 10k steps in `$WORK`, and after eval keep only
   final plus best per arm. Twenty arms x 2 kept milestones = ~210 GB in `$WORK`.
5. **Make walltime invisible.** Fix resume first (`save_state` + `StatefulDataLoader` +
   `completed_steps` + epoch counters), save every 20 to 40 minutes and on SIGUSR1, run
   with `--requeue` and `--signal=B:USR1@900`. Pin the node count for a campaign because
   ZeRO-2 shards optimizer state per rank.
6. **Everything that touches the network happens before `sbatch`.** Offline wheels or an
   Apptainer image built on Hetzner, base models pre-downloaded with `HF_HUB_OFFLINE=1`
   tested, `all_steps` pickle and dataset statistics warmed in a serial job, checksums
   verified against the HP manifest.

## Space budget

| Area | Contents | Estimate |
|---|---|---|
| `$SCRATCH` | LIBERO LeRobot, SSv2 webm (~20 GB), DROID re-encoded (200-400 GB), later raw RLDS transiently if transcoding on DCGP | ~0.5 TB steady, 2.2 TB transient |
| `$FAST` | Apptainer image or venv (~15 GB), Qwen3-VL-2B (~4.5 GB), V-JEPA 2 ViT-L (~1.2 GB), LeVJEPA-L (~1.2 GB), pretrain ckpt (~5 GB), rolling full-state saves: 93 GB per live job | ~30 GB + 93 GB x concurrent jobs |
| `$WORK` | code, milestones (5.2 GB each), tensorboard, eval videos/results, `all_steps` caches | ~250 GB at end of Phase 1 with pruning |
| `$HOME` | dotfiles only | < 5 GB |

Four concurrent Phase 1 jobs fit in `$FAST` with ~600 GB to spare. If `$SCRATCH` turns out
to carry the 20 TB figure the overview table shows, it still holds everything above.

## Task list

Ordered; each step is a gate for the next. Owner and location in brackets.

### A. Before touching Leonardo [Hetzner / HP / laptop]

1. Send the CINECA asks (superc@cineca.it): requeue permission, `$WORK` or `$DRES` quota
   increase, whether `lrd_all_serial` has outbound network, Globus endpoint, longer
   walltime. Lead time is long; ask now.
2. Fix resume in `train_jevla_cotrain.py`: `accelerator.save_state`/`load_state`,
   `StatefulDataLoader` (torchdata >= 0.8) for both loaders, persist `completed_steps` and
   both epoch counters, save on `STOP_AND_SAVE` sentinel file, raise `NCCL_TIMEOUT` to
   3600. Fix the double `lr_scheduler.step()` in the same change. Prove it on Hetzner or
   any GPU: train 200 steps, kill at 100, resume, LR and loss continuous.
3. Fix `_reset_dataloader` so the mixture's `dataset.epoch` actually advances (the brief
   flags that only `sampler.set_epoch` is called); otherwise every epoch replays the same
   sample order and the (epoch, index) resume shortcut is meaningless.
4. Build the runtime offline-capable: either `pip download` all wheels including a
   compiled `flash-attn`, or an Apptainer `.sif` built on Hetzner (x86_64, CUDA 12.x
   matching Leonardo drivers) containing the training env and, separately, the LIBERO and
   LIBERO-Plus eval envs with EGL. Test with `HF_HUB_OFFLINE=1` and no network.
5. Download to Hetzner: Qwen3-VL-2B-Instruct, `facebook/vjepa2-vitl-fpc64-256`,
   `galilai-group/LeVJEPA-VideoMix-Large`, `ginwind/VLA-JEPA` pretrain and LIBERO
   checkpoints, LIBERO LeRobot datasets, SSv2. Write a manifest with sha256 per file; keep
   the master copy on the HP box.
6. Start the DROID re-encode on Hetzner in the background (Phase 3 dependency only).
   Measure one episode first and extrapolate the wall time. Write the manifest as shards
   complete.

### B. First contact with Leonardo [login node / data mover]

7. Set up the account: 2FA, `step ssh login`, an SSH control master to
   `data.leonardo.cineca.it`. Record absolute paths for `$WORK`, `$FAST`, `$SCRATCH`
   (undefined on data movers).
8. Run the two-minute egress test from a `boost_qos_dbg` allocation (DNS and `curl`
   separately). Design assumes no egress either way, but a stray online call must fail
   fast, not hang holding 32 GPUs.
9. Push Phase 0-2 inputs (models, checkpoints, LIBERO data, SSv2, runtime image) with
   `rsync --partial --append-verify` or GridFTP. Verify checksums against the manifest on
   Lustre before anything else.
10. Warm the caches in a serial job: `all_steps` pickle and `save_dataset_statistics` for
    each dataset mix. Store under `$WORK/cache`.
11. Set environment defaults in the job template: `HF_HOME`, `TORCH_HOME` -> `$FAST`;
    `TMPDIR` -> `$SCRATCH`; `HF_HUB_OFFLINE=1`; `WANDB_MODE=offline`; drop
    `NCCL_IB_DISABLE=1`, set `NCCL_SOCKET_IFNAME=ib0`, `NCCL_IB_HCA=mlx5`,
    `NCCL_NET_GDR_LEVEL=5`; `num_workers` ~7 per GPU; `FFMPEG_THREADS=1`.

### C. Phase 0 on one node [Booster, boost_usr_prod]

12. 200-step smoke test, one node, per-device batch 8, gradient checkpointing on. Record
    step time, peak memory on 64 GB, dataloader throughput. Re-derive every cost figure in
    `EXPERIMENT_BED.md` from the measured step time.
13. Resume seam test on Leonardo itself: submit with `--time=00:30:00`,
    `--signal=B:USR1@300`, `--requeue`; confirm the second job continues LR, loss and
    dataloader position.
14. Run the diagnostics on the released LIBERO checkpoint (z noise/shuffle/drop, CKA to
    frozen VLM, direction probe) to establish the baseline the sweep is judged against.
15. Measure the anchor teacher's memory cost with batch 8 to fix the Anchor-Align arm's
    batch size and accumulation.

### D. Phase 1 sweep [Booster, boost_qos_lprod or boost_usr_prod]

16. One config per arm under `scripts/configs/research/`, all reading the same staged
    corpus; the sweep varies code and config, never data.
17. Job template: rolling saves to `$FAST/ckpt/<run>/` (keep 3), milestone export of
    bf16 weights to `$WORK/milestones/<run>/` every 10k steps, tensorboard to `$WORK`.
18. Eval job template: LIBERO then LIBERO-Plus from the Apptainer eval image, results to
    `$WORK/results/<run>/`. Prune milestones to final plus best after eval.
19. Weekly housekeeping: `find $SCRATCH -type f -exec touch -a` (or read the tree) to
    refresh the 40-day access clock while a phase is idle; check `cindata`.

### E. Phase 3 [after the DROID re-encode lands]

20. Push DROID re-encoded shards to `$SCRATCH/droid256/`, verify checksums, warm its
    `all_steps` cache and statistics. Fix the SSv2 caption source at the same time.
21. Pretraining on 8 nodes x 4 GPUs in 4-day chunks with the requeue chain; two configs.

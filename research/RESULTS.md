# Results log

Newest first. Raw outputs under `research/results/`. Every entry names the checkpoint, the
data, the command, and what was concluded, so a conclusion can be re-derived later.

## 2026-09-11 — released LIBERO checkpoint: the world model does not read z

Checkpoint: `ginwind/VLA-JEPA` `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt` (paper's LIBERO model,
V-JEPA 2 clip targets, 3 transitions x 8 tokens). Data: 256 samples from
`libero_spatial_no_noops_1.0.0_lerobot` (val-mode sampling, seed 0). Machine: RTX 4070 laptop,
bf16, SDPA attention. Command: `research/diagnose.py --data_mix libero_spatial --num_batches 64
--batch_size 4` (see `research/results/released_libero/libero_spatial.md`).

World-model L1 on the target states:

| condition | L1 |
|---|---|
| true z | 1.1338 |
| z from another sample (same batch) | 1.1339 |
| z from another sample, different task | 1.1341 |
| z := batch mean of z | 1.1338 |
| z := running dataset mean of z | 1.1340 |
| z tokens permuted within the sample | 1.1624 |
| z := 0 | 1.3015 |
| copy last input state (no predictor) | 1.3983 |
| z := matched Gaussian noise | 1.5474 |
| targets from another clip (scene cut) | 1.7811 |

Paired per-sample differences (true minus shuffled / mean-z): mean 0.0000, mean absolute
0.0002, n = 256. Inside the predictor, `action_encoder(z)` has mean token norm 43,153 versus
691 for `predictor_embed(states)`.

Reading:

1. **z is functionally a constant.** Replacing every sample's latent-action tokens by the
   dataset-mean vector changes the world-model loss by 1e-4. The predictor is sensitive to z
   being present (zeros +0.17) and to token order (+0.03), i.e. it learned a per-slot bias
   from z, but nothing sample-specific gets through. The 60x norm ratio says why: the
   `<|action_i|>` hidden states are dominated by a shared huge component, and the per-sample
   variation is a rounding error on top of it.
2. **The world model works from context states, not from z.** True z beats copy-last by
   0.26, and scene-cut destroys the loss, so the predictor does use the input states; with
   bidirectional clip features, s_0 already carries future information (research question in
   `WHY_THIS_FORK.md` §3), which is consistent with z being redundant.
3. **z "encodes actions" only in the sense that every position does.** Ridge R² z -> normalised
   action chunk 0.90 (PCA-64: 0.84), direction probe 0.81 vs 0.34 chance; but the embodied
   tokens give 0.86 / 0.81 and the pre-action hidden state 0.74 / 0.79 on the same probes. In
   LIBERO the action chunk is largely a function of instruction + image, which the whole
   sequence carries.
4. Action MAE (normalised) 0.036 confirms the checkpoint and data pipeline are healthy.

**Replication on `libero_10`** (256 samples, `research/results/released_libero/libero_10.md`):
true z 1.1886, shuffle 1.1885, other-task shuffle 1.1877, batch-mean 1.1886, global-mean 1.1885,
zeros 1.3289, copy-last 1.3774, noise 1.5535, scene-cut 1.8740; paired |diff| 0.0001; norm ratio
43,108 vs 691. R² z -> actions 0.91 (embodied 0.89), direction probe 0.75 (embodied 0.79). Same
picture on the long-horizon suite.

Implication for the plan (`TASKS.md`): the `noz` control should match `base_30k` on the WM loss
almost exactly and the interesting question moves to (a) whether the WM loss still helps the
policy purely as a regulariser (base vs noz on LIBERO / LIBERO-Plus) and (b) whether leak-free
targets (`perframe`, `levjepa`) or a real bottleneck (`proj*`, `vib32`, `ln`) force sample
information through z. Pending: same diagnostics on `libero_10` (running) and on the Pretrain
checkpoint; z-geometry stats (mean-vector norm vs deviation norm, pairwise cosine) added to
`diagnose.py` for the next run.

# Results log

Newest first. Raw outputs under `research/results/`. Every entry names the checkpoint, the
data, the command, and what was concluded, so a conclusion can be re-derived later.

## 2026-09-14 — the encoder leak is total, and the world-model gradient to the VLM is 1/1600 of the action gradient

Checkpoint: released `LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`. Data: 64 `libero_spatial` samples
(val-mode, seed 0). Machine: RTX 4070 laptop, bf16. Command: `research/leak_and_gradient.py
--num_batches 16 --batch_size 4` (raw: `research/results/released_libero/leak_gradient.{json,md}`).

**Encoder leak** (relative change of each V-JEPA 2 state when frames are perturbed; the scale is
the per-state deviation across samples, so about 1.4 means "as different as another sample"):

| encoder | perturbation | s_0 | s_1 | s_2 | s_3 |
|---|---|---|---|---|---|
| clip (upstream) | future frames from another sample | 1.35 (cos 0.62) | 1.64 | 1.63 | 1.62 |
| clip (upstream) | future frames frozen at frame 1 | 1.35 (cos 0.63) | 1.41 | 1.49 | 1.56 |
| clip (upstream) | past frames from another sample | 1.62 | 1.14 | 0.92 | 0.97 |
| per-frame (same weights) | future frames from another sample | **0.000** | 0.000 | 1.64 | 1.64 |
| per-frame (same weights) | past frames from another sample | 1.64 | 1.64 | **0.000** | 0.000 |

(per-frame has 8 states; only the first four columns are shown, s_2.. are the perturbed ones.)
Replacing the frames *after* the present changes the "current" state s_0 as much as swapping the
whole sample. Even freezing the future (no motion after frame 1) moves s_0 by the same amount.
The predictor's context therefore contains the answer to the prediction task, and the L1 target
s_{k+1} contains frames beyond it as well. The per-frame encoder with identical weights has
exactly zero future leak and exactly zero past leak into later states, as it must.

**Gradient decomposition** (weighted world-model loss, `wm_loss_weight` 0.1, d L / d z per token;
z = m + r with m the per-slot batch mean and r the per-sample residual):

| quantity | value |
|---|---|
| ‖∂L_wm/∂z‖ per latent token | 7.6e-8 |
| ‖∂L_action/∂embodied‖ per embodied token | 1.7e-4 |
| ratio world-model : action gradient | **6.4e-4** (about 1 : 1,600) |
| \|⟨g, m⟩\| (loss change for a 100% rescale of the shared part) | 7.5e-7 |
| \|⟨g, r⟩\| (loss change for a 100% rescale of the residual) | 1.6e-7 |
| finite difference, residual ×2 | 1.3e-5 |
| finite difference, residual ×0 | 2.3e-7 |
| finite difference, shared ×1.1 | 5.5e-7 |
| \|cos(g, r)\| / \|cos(g, m)\| | 0.015 / 0.012 |
| ‖m‖ / ‖r‖ per token | 1164 / 282 |

Reading: the world-model loss is locally flat in every direction of z. A doubling of the
per-sample residual moves the weighted loss by 1e-5 (1e-4 unweighted, matching the shuffle
result of 11 Sep); a 10% change of the shared component moves it by 5e-7. The gradient is not
aligned with either the shared or the residual direction (cosines 0.01), i.e. what little there
is points nowhere useful. Per token, the VLM receives 1,600 times more gradient from the action
head than from the world model. The world-model term is therefore inert as a learning signal to
the VLM on this checkpoint, which is the mechanism behind the constant-z finding. Caveat: the
predictor runs in bf16 autocast, so the absolute gradient values carry bf16 noise; the
finite-difference rows are independent of that and agree.

Implementation landed with this entry: `center` / `center_ln` bottleneck kinds
(`latent_bottleneck.py`, EMA per-slot mean, stop-gradient, optional LayerNorm + gain), 7 unit
tests on a synthetic constant-dominated z, arm overlays `center`, `center_ln`, `perframe_center`.

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
information through z. 

**Pretrain checkpoint** (`Pretrain/checkpoints/VLA-JEPA-pretrain.pt`, DROID + SSv2 co-training,
evaluated on the same 256 `libero_spatial` samples, i.e. out of distribution;
`research/results/released_libero/pretrain_ckpt_libero_spatial.md`): true z 1.3864, shuffle
1.3864, other-task 1.3871, batch-mean 1.3863, global-mean 1.3864, zeros 1.4580, copy-last
1.3983, noise 1.4724, scene-cut 1.7180; paired |diff| 0.0006. Norm ratio inside the predictor
1,164 vs 63 (18x). Geometry of z: ||mean z|| 501 vs mean ||z - mean z|| 115, pairwise cosine
between samples 0.94 (embodied tokens 0.56). R² z -> LIBERO actions 0.45 (embodied 0.40).

So the constant-z regime is established during pretraining, not by the LIBERO fine-tune, and
on these clips the pretrained predictor beats copy-last by only 0.012. LIBERO fine-tuning makes
the per-sample part of z relatively *larger* (||mean|| 681 vs deviation 296, cosine 0.82) and
the predictor's amplification of z 3x stronger (43k vs 691), yet the world model still reads
none of it. Decomposing z inside the predictor (128 samples,
`research/results/released_libero/libero_spatial_decomposition.md`): the batch-constant part
goes 1,186 -> 43,676 through `action_encoder` (37x) and the per-sample residual 289 -> 2,897
(10x). The residual therefore survives the linear encoder, and is even 4x the norm of the
embedded video states, but it is a 6.6% perturbation of a token whose norm is dominated by the
constant. After the predictor's LayerNorms that token is essentially the normalised constant
direction, and the trained blocks are insensitive to the small angular change: the loss moves by
1e-4 when the residual is removed. The channel is not projected away by a single matrix; it is
drowned by a massive shared activation and ignored downstream.

Open: base vs `noz` on LIBERO / LIBERO-Plus (does the loss help as a regulariser?), and whether
leak-free targets or a real bottleneck put sample information into the channel (`perframe`,
`levjepa`, `proj*`, `vib32`, `ln`). A base-VLM CKA pass (needs a second 4.4 GB model, so not
on the 8 GB laptop) is still pending for the representation-drift question.

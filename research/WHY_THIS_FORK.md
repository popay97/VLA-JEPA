# Why this fork exists

Read this first. It records how we got from "understand VLA-JEPA" to the experiment bed in
`EXPERIMENT_BED.md`, so that anyone (human or agent) can pick up the thread without
re-deriving it. Each section states the question, the evidence, and the conclusion we
acted on. Code references are to upstream `main` at b49b016.

Owner: Vuk Stajkic (GitHub popay97). Compute: 5000 A100-hours on EuroHPC. Goal: characterize
the latent-action bottleneck in VLA-JEPA (arXiv:2602.10098v2), i.e. whether the latent
action tokens z actually carry action-relevant information and what design choices make
them do so.

## 0. Method note

Every paper referenced below was read in full text (arXiv HTML converted locally), not
from abstracts or summaries. The paper texts live outside the repo; arXiv IDs are given so
they can be re-fetched.

## 1. What VLA-JEPA actually is (paper vs code)

Question: what does the model do, and which design choices are arbitrary or undocumented?

Findings from `starVLA/model/framework/VLA_JEPA.py`, `vj2_predictor.py`, `vj2_modules.py`,
the YAML configs and dataloaders:

- Qwen3-VL-2B VLM (28 layers, H=2048) + frozen V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`,
  tubelet 2, patch 16, 256 px, 256 tokens per temporal step, D=1024) + randomly initialised
  V-JEPA2-AC-style predictor (12 layers, 8 heads, 3D RoPE, block time-causal mask) + GR00T
  N1.5 DiT flow-matching head (DiT-B, 16 layers, cross-attention dim 2048).
- Latent action tokens: 3 distinct `<|action_i|>` tokens repeated 8 times each = 24. The 3
  comes from `num_frames // tubelet_size - 1` (8 frames, tubelet 2, so 4 states and 3
  transitions). The 8 is `num_action_tokens_per_timestep`. The paper describes this
  inverted ("K=24, T=3 per step"). 28 tokens are added to the vocabulary
  (`action_horizon*4`), 25 unused.
- 32 `<|embodied_action|>` tokens feed the DiT through cross-attention.
- Frame flow: 8 raw frames f0..f7 per view -> encoder -> 4 states s0..s3. Two views are
  concatenated channel-wise (2048-d). Predictor input is [z_k(8) | s_k(256)] for k=0..2,
  outputs at s_k positions predict s_{k+1}, L1 loss against the frozen encoder. The VLM
  sees only raw f0 per view at 224x224.
- The "concat 2 frames" trick is really three separate 2-way packings: tubelet 2 in the
  encoder, two-view channel concat, and single-view datasets duplicated into a fake second
  view (`video_datasets.py`, `video=np.stack([video, video.copy()])`).
- No bottleneck on z. z goes through one `nn.Linear(2048, 1024, bias=True)`
  (`vj2_predictor.py:57`, `action_encoder`). No LayerNorm, nonlinearity, VQ, or KL. The
  only bottleneck is observational (VLM sees f0 + text), and the predictor is teacher-forced
  with the true s0..s2, so it can largely ignore z.
- Training defects: co-training alternates optimizer steps on a Droid batch and an SSv2
  batch and calls `lr_scheduler.step()` twice per iteration
  (`train_jevla_cotrain.py`, `_train_step`), so `cosine_with_min_lr` reaches its minimum at
  half of `max_train_steps` and rises again. World-model weight 0.1 is hard-coded
  (`VLA_JEPA.py:243`); `loss_scale` in YAML is dead. Flow-matching timesteps are
  Beta(1.5, 1.0), the paper says uniform.
- Data defects: SSv2 captions come from `test-answers.csv`, so most videos get the
  placeholder "Completing something that humans might want to do."; consecutive frames, no
  stride; no visual augmentation; actions min-max normalized on xyz+rpy, gripper raw.
- Eval: LIBERO executes the full 7-action chunk open-loop with images rotated 180 degrees;
  SimplerEnv re-queries each step with adaptive ensembling; LIBERO-Plus uses 1 trial per
  task.

Conclusion: the interesting scientific object is z, and the code gives z no reason to be
informative. Everything else in the bed follows from testing that.

## 2. Related work: what to expect and how to measure

Question: what have other latent-action works learned that narrows our search?

- Capacity sweet spot is small. Wayve LA-Pose: a 50-d latent beats 1536-d for pose even
  though the larger one reconstructs video better. Garrido et al. (arXiv:2601.05230):
  mid-capacity best, VQ fails in the wild, scene-cut test exposes leakage. LAWM
  (arXiv:2509.18428): 7-d latent, CCA to true actions ~0.9.
- Teacher-forced next-step prediction rewards copying. LingBot-VA 2.0 (arXiv:2607.08639)
  and Zero-WAM (arXiv:2608.26103) use strided future chunks (IFP raised 28.6 -> 47.0 in
  Zero-WAM). V-JEPA2-AC adds an autoregressive rollout loss.
- Measure z-dependence directly. LAWA (arXiv:2608.24882): noise injection into z dropped
  success 80.8 -> 52.2 when z mattered. Use noise, shuffle, and drop-z controls.
- Cost anchors: Garrido's run was 12 h on 64 H100; Zero-WAM pretraining 15,360 GPU-h. Our
  estimates (35% MFU, 2x overhead): VLA-JEPA pretrain B=8 50k steps ~180 A100-h, LIBERO
  fine-tune 120k ~210, 30k ~55, LIBERO eval ~10, LIBERO-Plus ~40-60.

Conclusion: sweep bottleneck capacity (LayerNorm, projection d in {8, 32, 128}, KL, VQ,
no-z control), add a strided-future and a rollout arm, and make z-dependence diagnostics
mandatory on every checkpoint.

## 3. The bidirectional-encoder leakage (Vuk's observation, confirmed)

Question: V-JEPA 2 is bidirectional, so is the world-model side leaking the future?

Evidence: HF `transformers` 4.57.0 `modeling_vjepa2.py`: `VJEPA2Encoder.forward` calls
every layer with attention mask `None`, and `VJEPA2SelfAttention.is_causal = False`.
VLA-JEPA calls `get_vision_features` on the whole 8-frame clip. So each state s_k contains
information from f0..f7, and both predictor inputs and L1 targets carry the future.

Contrast: Meta's V-JEPA2-AC training script (`app/vjepa_droid/train.py`, `forward_target`)
encodes each frame independently by duplicating it into a 2-frame tubelet and layer-norms
the targets. Garrido et al. use a frame-causal encoder. VLA-JEPA dropped both when porting
the predictor.

Conclusion: the paper's "leakage-free" claim holds only for the VLM branch. The world-model
branch violates it by construction, which also makes ignoring z even easier. This became
the encoder axis in the bed.

## 4. Replacement encoders

- LeVJEPA (arXiv:2608.27395, github.com/MLO-lab/LeVJEPA): block-causal attention in the
  encoder at no accuracy cost (51.2 vs 50.7 in their controlled setting), tubelet 1 matches
  or beats tubelet 2 even on SSv2, patch tokens show semantic PCA structure without a
  dense loss. Checkpoint `galilai-group/LeVJEPA-VideoMix-Large`: ViT-L/16, 303M, 16 frames
  at 224, output (1, 3137, 1024) = CLS + 16x14x14, ImageNet mean/std, keep
  `attn_mode="block_causal"`, `trust_remote_code=True`, weights CC BY-NC 4.0. Caveats: dense
  sufficiency unevaluated in the paper; 55.0 SSv2 probe vs 76.5 for distilled V-JEPA 2.1-L
  (much smaller corpus); grid change 14x14 at 224 vs current 16x16 at 256.
- V-JEPA 2.1 (arXiv:2603.14482): recovers dense features (V-JEPA 2's are "noisy,
  fragmented"), +20% grasp over V-JEPA2-AC, but the encoder is still bidirectional and
  weights are CC BY-NC-ND 4.0. Using it requires Meta's per-frame trick.

Conclusion: encoder axis = {V-JEPA 2 clip-level (current, leaky), V-JEPA 2 per-frame,
LeVJEPA block-causal}. Tubelet 1 gives 7 transitions, matching the 7-action chunk, which
removes one of the arbitrary mismatches from section 1.

## 5. Anchor-Align as a representation-preservation arm

Question: Vuk wanted Anchor-Align (arXiv:2607.13429) alongside latent actions. How does it
fit, and do we submodule, copy, or fork?

Findings: two losses, no new data. Anchor = frozen VLM copy, per-layer MSE on vision and
text positions. Align = direction word from chunk-averaged xyz, predicted from the last
text token through a learned projection and the frozen lm_head. On VLA-Adapter, LIBERO-PRO
61.0 -> 71.9, position swap 2.3 -> 22.6, LIBERO-Plus 85.1 -> 90.3; text-token CKA to the
pretrained VLM 0.34 -> 0.91. The repo's usable implementation is the starVLA snapshot under
`real_world_training/` (QwenGR00T + GR00T DiT, same stack as ours), about 150 lines.

Why it matters here: the world-model loss reaches the VLM only through z. Without an
anchor the backbone can absorb that signal by reshaping vision/text tokens. Anchoring
closes that route and makes z the only free channel, so Anchor doubles as a diagnostic for
where the world-model signal is going. Align is orthogonal to z and cheap.

Packaging decision: port, with attribution. A submodule is unimportable (their snapshot
is a full starVLA copy with the same package name), and a fork gains nothing. Details and
line-level wiring in `ANCHOR_ALIGN_INTEGRATION.md`; provenance in `THIRD_PARTY_NOTICES.md`.

## 6. Open questions the bed is designed to answer

1. Does z carry anything today? (noise/shuffle/drop tests on the released checkpoint)
2. Which bottleneck makes z informative without hurting success?
3. Does removing encoder leakage change z-dependence?
4. Does strided or rolled-out prediction beat teacher-forced next state?
5. Does anchoring raise z-dependence, and does it preserve GQA/CKA while keeping success?
6. Is 3 transitions x 8 tokens better or worse than 7 x 1?

## 7. Things not in the repo

- Full paper texts and the HF/Meta source files used as evidence were kept in a local
  scratch directory; re-fetch by arXiv ID or from the GitHub URLs in
  `THIRD_PARTY_NOTICES.md`.
- No GPU or torch on the authoring machine, so nothing in `research/` has been executed.
  First real run is Phase 0 in `EXPERIMENT_BED.md`.

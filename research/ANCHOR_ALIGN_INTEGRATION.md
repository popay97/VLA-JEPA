# Anchor-Align inside VLA-JEPA: integration map

Source: Dalal et al., "Generalizable VLA Finetuning via Representation Anchoring and
Language-Action Alignment", arXiv:2607.13429 (CC BY 4.0), code
https://github.com/dwipddalal/Anchor-Align (MIT). The relevant implementation is NOT the
main `prismatic/` tree (a VLA-Adapter fork) but the starVLA snapshot under
`real_world_training/starVLA/model/framework/VLM4A/QwenGR00T.py`, which adds the two losses to
the same Qwen-VL + GR00T DiT stack that VLA-JEPA uses.

## 1. What Anchor-Align is (two losses, no new data)

    L_total = L_action + 0.1 * L_anchor + 0.02 * L_align

**Anchor.** A frozen deep copy of the pretrained VLM runs the same batch. Per decoder
layer u, MSE between student and teacher hidden states on the vision + text positions
(action-token positions masked out), averaged over layers, scaled by 1/(2 sigma^2), sigma=1.
The teacher's `lm_head` is replaced by `nn.Identity()`; only hidden states are needed.
Reported overhead: +28% wall clock, +0.7 GB on a 0.5B backbone; sublinear for larger VLMs.

**Align.** Take the last-layer hidden state of the last text token before the action
tokens, project through a learned `Linear(H, H)`, then through the frozen `lm_head`, and
cross-entropy against one of six single-token direction words derived from the
chunk-averaged xyz action delta (dominant axis + sign). Samples with ||mean xyz||_2 below
a threshold are ignored (-100). Shuffle/Scatter controls in the paper show the gain is not
generic regularization.

Reported effect (VLA-Adapter, LIBERO-Spatial): LIBERO-PRO mean 61.0 -> 71.9, position swap
2.3 -> 22.6, LIBERO-Plus 85.1 -> 90.3; text-token CKA to the pretrained VLM 0.34 -> 0.91.
Ablation: anchor-only 68.1/87.3, align-only 65.9/88.6, both 71.9/90.3.

## 2. Why it matters for the latent-action bottleneck question

VLA-JEPA gives the world-model loss a gradient path into the VLM only through the 24
`<|action_i|>` positions (z). Nothing stops the VLM from instead reshaping the vision/text
token representations to make the predictor's job easier, since the predictor is
teacher-forced with the true states anyway. Anchoring the vision/text positions removes
that escape route: z becomes the only unconstrained channel through which VLM-side
information can reach the predictor. So Anchor is both a preservation mechanism and a
tool for the bottleneck study:

- If z-dependence (noise/shuffle test, CCA to actions) rises under anchoring, the current
  weak dependence is partly because the backbone was absorbing the WM signal elsewhere.
- If task success holds while GQA/CKA stay high, the WM objective is compatible with
  preserving the VLM, which is the paper's implicit but untested claim.
- The paper's own diagnostics (layer-wise text-token CKA, linear-probe R^2 for actions,
  direction-word probe) slot directly into the Phase 0 diagnostic suite.

Align is orthogonal to z: it supervises the last text token, not the action tokens. It is
worth one arm because it is cheap and because the paper shows it makes action information
linearly decodable in the backbone (R^2 0.60 at layer 22), which is exactly the quantity a
latent action should carry.

## 3. Where it plugs into this repo

All line numbers refer to `starVLA/model/framework/VLA_JEPA.py` on `main` (b49b016).

| Step | VLA-JEPA site | What changes |
|---|---|---|
| Teacher | `__init__`, after `expand_tokenizer` (line 63) | `copy.deepcopy(self.qwen_vl_interface)`, freeze, `.eval()`, `model.lm_head = nn.Identity()`. Copy AFTER `resize_token_embeddings` (line 127) so both models accept the 28 new token ids. |
| Student hidden states | `forward`, line 170-177 | Already `output_hidden_states=True`; keep `qwenvl_outputs.hidden_states[1:]` (28 layers for Qwen3-VL-2B, H=2048) instead of only `[-1]`. |
| Teacher forward | right after the student forward | `torch.no_grad()` + bf16 autocast on the same `qwen_inputs`. |
| Anchor mask | lines 163-167 | `keep = attention_mask & ~isin(input_ids, action_token_ids) & ~isin(input_ids, [embodied_action_token_id])`. Padding is left-side (`QWen3.py:65`), so the mask must come from `attention_mask`, not from position. |
| Anchor loss | new | mean over 28 layers of `0.5/sigma^2 * mse(student[keep], teacher[keep])`, computed in fp32. |
| Align position | lines 163-164 | first `<|action_0|>` index per row minus one. Do NOT use "last non-pad position": in VLA-JEPA the sequence ends with 32 `<|embodied_action|>` tokens plus the generation prompt. |
| Align labels | `forward`, `actions` (line 142) | actions are already EEF deltas (LIBERO/Droid: xyz, rpy, gripper), min-max normalized to [-1, 1] on the six pose dims. No FK cache needed (the xArm7 version needed one because it trained in joint space). Threshold must be re-tuned for normalized units (paper 0.15 in VLA-Adapter units, xArm7 0.0006 m). |
| Align head | `__init__` | `nn.Linear(2048, 2048)` + frozen `lm_head.weight.detach()`. Qwen3-VL-2B ties embeddings, so `lm_head.weight` is the (resized) input embedding matrix. |
| Loss dict | line 218 and 243 | add `"anchor_loss"` and `"align_loss"` entries. `train_jevla_cotrain.py:411` sums all dict values, so weights must be applied inside `forward` (same pattern as the hard-coded `* 0.1` on `wm_loss`). |
| Video batch | line 218 branch | anchor should also run on the SSv2 batch (the VLM is updated by that step too); align cannot (no actions). |
| Logging | `train_jevla_cotrain.py:460-463` | already logs every key as `vla_<k>` / `vlm_<k>`; nothing to add. |
| Inference | `predict_action` | untouched; teacher and align head are training-only. Exclude the teacher from checkpoints (`state_dict` filter) or checkpoints double in size. |

## 4. Design decisions to make before coding

1. **Anchor set.** Vision + text positions only (paper default). Anchoring the 24 latent
   action positions would pin z to the frozen VLM's output for a random new embedding,
   which defeats the world model. Anchoring the 32 embodied positions would similarly fight
   the action head.
2. **Anchor scope in pretraining.** VLA-JEPA trains the full VLM at lr 1e-5 (no LoRA), on
   Droid + SSv2 for 50k steps. Drift is therefore larger than in the paper's 10k-step LoRA
   setting; lambda_anchor may need to be higher than 0.1, or applied to a subset of layers.
   Start at 0.1, log per-layer CKA every save interval.
3. **Layer set.** All 28 (paper: "anchoring the full stack is the single most important
   design choice"). Keep `mse_layers: all | last` as a config switch for the ablation.
4. **Align threshold.** Compute the distribution of ||mean xyz|| over the LIBERO/Droid
   training set at Phase 0 and set the threshold at the ~20th percentile (the paper's 0.15
   removes near-stationary chunks, not a fixed fraction).
5. **Direction convention.** LIBERO actions are in the robot base frame; the paper's
   `run_alignment_test.py` header notes their spatial/object/goal checkpoints used a
   transposed x/y convention. Fix one mapping (x -> forward/backward, y -> left/right,
   z -> up/down) and verify it on a few demos by eye before training.
6. **Memory.** Frozen Qwen3-VL-2B in bf16 is ~4.4 GB per GPU plus its activations for one
   forward. With per-device batch 8 and gradient checkpointing this fits on 40 GB A100; on
   the 80 GB nodes it is a non-issue.

## 5. Packaging decision

Port, do not submodule and do not fork.

- Anchor-Align's main tree is a VLA-Adapter/OpenVLA (RLDS, Prismatic) codebase we will
  never import. Its starVLA snapshot is a full copy of starVLA at commit 5ce9e2e with a
  colliding `starVLA` package name, so a submodule would be unimportable next to ours.
- The actual logic is ~150 lines in one framework file plus ~20 lines of trainer logging
  and a dataset hook we do not need (FK cache).
- MIT permits copying with attribution. Record the upstream URL, commit, and file in
  `THIRD_PARTY_NOTICES.md` and in the header of the ported module.

Target location: `starVLA/model/modules/regularizers/anchor_align.py` with pure functions
(`masked_layerwise_mse`, `direction_labels`, `align_ce`) and a thin
`FrozenTeacher` wrapper, wired into `VLA_JEPA.py` behind `framework.anchor_align.*` flags
that default to off so the baseline is byte-identical.

## 6. Config surface (proposed)

    framework:
      anchor_align:
        enable_anchor: false
        anchor_weight: 0.1
        anchor_sigma: 1.0
        anchor_layers: all          # all | last
        anchor_on_video_batch: true
        enable_align: false
        align_weight: 0.02
        align_min_xyz_norm: 0.0     # set from Phase 0 statistics
        align_axis_map: [x, y, z]   # forward/backward, left/right, up/down

## Sources

- Paper: https://arxiv.org/abs/2607.13429
- Code: https://github.com/dwipddalal/Anchor-Align (starVLA port under `real_world_training/`)
- Checkpoints: https://huggingface.co/Dwipz/Anchor-Align

# Third-party notices

This fork ports or depends on the following external work. Ported code carries a header
naming its origin; this file is the index.

## Anchor-Align (MIT)

- Repository: https://github.com/dwipddalal/Anchor-Align
- Paper: Dalal et al., arXiv:2607.13429 (CC BY 4.0)
- Ported from: `real_world_training/starVLA/model/framework/VLM4A/QwenGR00T.py`
  (frozen-teacher layer-wise MSE anchoring, alignment-v7 direction-word loss) and
  `real_world_training/starVLA/training/train_starvla.py` (sub-loss logging).
  Their snapshot derives from starVLA commit 5ce9e2ed59dc94de926c335d36a3681c243a58f7.
- Destination: `starVLA/model/modules/regularizers/anchor_align.py` (to be added).
- Status: not yet ported; see `research/ANCHOR_ALIGN_INTEGRATION.md`.

## LeVJEPA (code MIT-style per repo; weights CC BY-NC 4.0)

- Repository: https://github.com/MLO-lab/LeVJEPA
- Paper: arXiv:2608.27395
- Used as: HF checkpoint `galilai-group/LeVJEPA-VideoMix-Large` via `trust_remote_code`,
  pinned to revision `e831a0347737fcaa660b39c57d41c109de399845` (2026-09-14) in the arm configs.
  The GitHub repo (MIT, training code with its own Lightning/Hydra stack) is not vendored or
  submoduled: inference needs only the two modeling files that ship inside the HF snapshot.
  Non-commercial weight license; research use only.

## V-JEPA 2 / V-JEPA 2.1 / V-JEPA2-AC (Meta)

- Repository: https://github.com/facebookresearch/vjepa2
- Papers: arXiv:2506.09985, arXiv:2603.14482
- Used as: HF checkpoint `facebook/vjepa2-vitl-fpc64-256` (already in upstream);
  `forward_target` per-frame encoding pattern from `app/vjepa_droid/train.py` to be
  reimplemented for the per-frame encoder arm. V-JEPA 2.1 weights are CC BY-NC-ND 4.0.

## Upstream

- VLA-JEPA: https://github.com/ginwind/VLA-JEPA (no LICENSE file at fork time; README
  badge Apache-2.0, pyproject MIT).
- starVLA: https://github.com/starVLA/starVLA (MIT).

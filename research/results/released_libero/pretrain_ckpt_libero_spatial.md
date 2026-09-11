# Diagnostics: `C:/Users/vukst/vlajepa_local/models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt`

encoder `vjepa2_clip`, bottleneck `none`, 256 samples, 3 transitions

## World-model L1 (lower is better)

| condition | mean | std |
|---|---|---|
| z_true | 1.3864 | 0.0198 |
| z_zeros | 1.4580 | 0.0207 |
| z_noise | 1.4724 | 0.0211 |
| z_shuffle | 1.3864 | 0.0198 |
| z_shuffle_other_task | 1.3871 | 0.0218 |
| z_batchmean | 1.3863 | 0.0198 |
| z_globalmean | 1.3864 | 0.0200 |
| z_tokshuffle | 1.3989 | 0.0230 |
| scene_cut | 1.7180 | 0.0383 |
| copy_last | 1.3983 | 0.0174 |

z effect: zeros-true +0.0716, shuffle-true +0.0000, copy_last-true +0.0119

- paired per-sample shuffle_minus_true: mean +0.0000, |diff| mean 0.0006, std 0.0008 (n=256)
- paired per-sample batchmean_minus_true: mean -0.0001, |diff| mean 0.0004, std 0.0005 (n=256)
- paired per-sample globalmean_minus_true: mean -0.0002, |diff| mean 0.0004, std 0.0005 (n=252)
- predictor input norms: action_encoder(z) 1164.21 vs predictor_embed(states) 63.03

## z -> action chunk

- ridge R^2 (PCA 64, evr 0.79): **0.383**; full-dim 0.454; xyz only 0.394
- CCA top-8: 0.95, 0.92, 0.85, 0.85, 0.80, 0.71, 0.68, 0.66
- effective rank of z: 71.3; ||mean z|| 500.7 vs mean ||z - mean z|| 115.4; pairwise cosine z 0.941, embodied 0.562
- embodied tokens -> actions R^2: 0.404; pre-action hidden -> actions R^2: 0.156

## Direction probe (6-way)

- labels [52, 8, 48, 12, 49, 87] (valid 1.00)
- from z: acc 0.578 (chance 0.340)
- from pre-action hidden: acc 0.484
- from embodied: acc 0.535

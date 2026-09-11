# Diagnostics: `C:/Users/vukst/vlajepa_local/models/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`

encoder `vjepa2_clip`, bottleneck `none`, 256 samples, 3 transitions

## World-model L1 (lower is better)

| condition | mean | std |
|---|---|---|
| z_true | 1.1338 | 0.0251 |
| z_zeros | 1.3015 | 0.0232 |
| z_noise | 1.5582 | 0.0809 |
| z_shuffle | 1.1339 | 0.0251 |
| z_shuffle_other_task | 1.1341 | 0.0281 |
| z_batchmean | 1.1338 | 0.0251 |
| z_globalmean | 1.1340 | 0.0252 |
| z_tokshuffle | 1.1624 | 0.0328 |
| scene_cut | 1.7811 | 0.0506 |
| copy_last | 1.3983 | 0.0174 |

z effect: zeros-true +0.1676, shuffle-true +0.0000, copy_last-true +0.2645

- paired per-sample shuffle_minus_true: mean +0.0000, |diff| mean 0.0002, std 0.0002 (n=256)
- paired per-sample batchmean_minus_true: mean -0.0000, |diff| mean 0.0001, std 0.0002 (n=256)
- paired per-sample globalmean_minus_true: mean -0.0000, |diff| mean 0.0001, std 0.0002 (n=252)
- predictor input norms: action_encoder(z) 43153.34 vs predictor_embed(states) 691.47

## z -> action chunk

- ridge R^2 (PCA 64, evr 0.76): **0.839**; full-dim 0.903; xyz only 0.872
- CCA top-8: 0.99, 0.99, 0.99, 0.98, 0.97, 0.97, 0.94, 0.94
- effective rank of z: 85.6; ||mean z|| 681.0 vs mean ||z - mean z|| 296.2; pairwise cosine z 0.819, embodied 0.214
- embodied tokens -> actions R^2: 0.864; pre-action hidden -> actions R^2: 0.738

## Direction probe (6-way)

- labels [52, 8, 48, 12, 49, 87] (valid 1.00)
- from z: acc 0.809 (chance 0.340)
- from pre-action hidden: acc 0.785
- from embodied: acc 0.812

## Action MAE (normalised): 0.0364

# Diagnostics: `C:/Users/vukst/vlajepa_local/models/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`

encoder `vjepa2_clip`, bottleneck `none`, 128 samples, 3 transitions

## World-model L1 (lower is better)

| condition | mean | std |
|---|---|---|
| z_true | 1.1245 | 0.0258 |
| z_zeros | 1.2923 | 0.0255 |
| z_noise | 1.5466 | 0.0590 |
| z_shuffle | 1.1245 | 0.0259 |
| z_shuffle_other_task | 1.1253 | 0.0253 |
| z_batchmean | 1.1245 | 0.0259 |
| z_globalmean | 1.1252 | 0.0260 |
| z_tokshuffle | 1.1486 | 0.0322 |
| scene_cut | 1.7749 | 0.0495 |
| copy_last | 1.3913 | 0.0200 |

z effect: zeros-true +0.1679, shuffle-true +0.0000, copy_last-true +0.2669

- paired per-sample shuffle_minus_true: mean +0.0000, |diff| mean 0.0001, std 0.0002 (n=128)
- paired per-sample batchmean_minus_true: mean -0.0000, |diff| mean 0.0001, std 0.0002 (n=128)
- paired per-sample globalmean_minus_true: mean -0.0000, |diff| mean 0.0001, std 0.0002 (n=124)
- predictor input norms: action_encoder(z) 43712.61 vs predictor_embed(states) 692.17
- z decomposition (raw -> after action_encoder): batch-mean part 1186.0 -> 43675.8; per-sample residual 289.2 -> 2896.6

## z -> action chunk

- ridge R^2 (PCA 64, evr 0.87): **0.428**; full-dim 0.885; xyz only 0.657
- CCA top-8: 1.00, 1.00, 0.99, 0.99, 0.98, 0.97, 0.97, 0.95
- effective rank of z: 52.8; ||mean z|| 691.9 vs mean ||z - mean z|| 309.4; pairwise cosine z 0.814, embodied 0.211
- embodied tokens -> actions R^2: 0.834; pre-action hidden -> actions R^2: 0.223

## Direction probe (6-way)

- labels [25, 4, 22, 6, 24, 47] (valid 1.00)
- from z: acc 0.602 (chance 0.367)
- from pre-action hidden: acc 0.602
- from embodied: acc 0.617

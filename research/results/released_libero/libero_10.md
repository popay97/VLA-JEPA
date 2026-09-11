# Diagnostics: `C:/Users/vukst/vlajepa_local/models/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`

encoder `vjepa2_clip`, bottleneck `none`, 256 samples, 3 transitions

## World-model L1 (lower is better)

| condition | mean | std |
|---|---|---|
| z_true | 1.1886 | 0.0197 |
| z_zeros | 1.3289 | 0.0185 |
| z_noise | 1.5535 | 0.0822 |
| z_shuffle | 1.1885 | 0.0197 |
| z_shuffle_other_task | 1.1877 | 0.0192 |
| z_batchmean | 1.1886 | 0.0197 |
| z_globalmean | 1.1885 | 0.0199 |
| z_tokshuffle | 1.2127 | 0.0262 |
| scene_cut | 1.8740 | 0.0431 |
| copy_last | 1.3774 | 0.0145 |

z effect: zeros-true +0.1403, shuffle-true -0.0000, copy_last-true +0.1888

- paired per-sample shuffle_minus_true: mean -0.0000, |diff| mean 0.0001, std 0.0002 (n=256)
- paired per-sample batchmean_minus_true: mean -0.0000, |diff| mean 0.0001, std 0.0001 (n=256)
- paired per-sample globalmean_minus_true: mean +0.0000, |diff| mean 0.0001, std 0.0001 (n=252)
- predictor input norms: action_encoder(z) 43108.23 vs predictor_embed(states) 691.38

## z -> action chunk

- ridge R^2 (PCA 64, evr 0.75): **0.865**; full-dim 0.906; xyz only 0.886
- CCA top-8: 0.99, 0.99, 0.98, 0.98, 0.98, 0.95, 0.94, 0.87
- effective rank of z: 83.7
- embodied tokens -> actions R^2: 0.892; pre-action hidden -> actions R^2: 0.709

## Direction probe (6-way)

- labels [38, 22, 43, 32, 40, 81] (valid 1.00)
- from z: acc 0.754 (chance 0.316)
- from pre-action hidden: acc 0.695
- from embodied: acc 0.785

## Action MAE (normalised): 0.0231

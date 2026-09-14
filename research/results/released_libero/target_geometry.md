# Target-encoder leak and geometry (libero_spatial, 64 samples)

## Leak: relative change of each state (0 = unchanged, ~1.4 = as different as another sample)

### `clip` (vjepa2_clip, 4 states)

| perturbation | s_0 | s_1 | s_2 | s_3 |
|---|---|---|---|---|
| future_swap rel | 1.349 | 1.641 | 1.625 | 1.620 |
| future_frozen rel | 1.354 | 1.415 | 1.489 | 1.562 |
| past_swap rel | 1.623 | 1.142 | 0.918 | 0.965 |

### `perframe` (vjepa2_perframe, 8 states)

| perturbation | s_0 | s_1 | s_2 | s_3 | s_4 | s_5 | s_6 | s_7 |
|---|---|---|---|---|---|---|---|---|
| future_swap rel | 0.000 | 0.000 | 1.640 | 1.639 | 1.640 | 1.640 | 1.638 | 1.637 |
| future_frozen rel | 0.000 | 0.000 | 1.082 | 1.217 | 1.258 | 1.285 | 1.314 | 1.324 |
| past_swap rel | 1.641 | 1.641 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### `levjepa` (levjepa, 8 states)

| perturbation | s_0 | s_1 | s_2 | s_3 | s_4 | s_5 | s_6 | s_7 |
|---|---|---|---|---|---|---|---|---|
| future_swap rel | 0.000 | 0.000 | 1.809 | 1.763 | 1.809 | 1.784 | 1.735 | 1.702 |
| future_frozen rel | 0.000 | 0.000 | 1.350 | 2.289 | 2.376 | 2.292 | 2.030 | 1.882 |
| past_swap rel | 1.644 | 1.624 | 1.557 | 1.306 | 1.186 | 1.014 | 0.858 | 0.752 |

## Geometry of the target tokens

### raw

| quantity | `clip` | `perframe` | `levjepa` |
|---|---|---|---|
| mean_vector_norm | 49.34 | 50.87 | 28.19 |
| deviation_norm_mean | 79.73 | 77.91 | 9.441 |
| mean_to_deviation_ratio | 0.6188 | 0.6529 | 2.986 |
| energy_fraction_of_mean_direction | 0.2758 | 0.2976 | 0.8952 |
| pairwise_cosine_cross_sample | 0.2732 | 0.2961 | 0.895 |
| pc1_variance_fraction_after_centering | 0.1023 | 0.1211 | 0.1021 |
| effective_rank_after_centering | 144.9 | 130.3 | 110.6 |
| massive_channels | 1 | 1 | 12 |
| top_channel_abs_mean_over_median_std | 16.39 | 17.98 | 74.46 |

### after_layernorm

| quantity | `clip` | `perframe` | `levjepa` |
|---|---|---|---|
| mean_vector_norm | 16.83 | 17.5 | 30.28 |
| deviation_norm_mean | 27.18 | 26.76 | 10.14 |
| mean_to_deviation_ratio | 0.619 | 0.6541 | 2.988 |
| energy_fraction_of_mean_direction | 0.2765 | 0.2992 | 0.8956 |
| pairwise_cosine_cross_sample | 0.2732 | 0.296 | 0.895 |
| pc1_variance_fraction_after_centering | 0.09945 | 0.1177 | 0.1029 |
| effective_rank_after_centering | 145.9 | 131.8 | 111 |
| massive_channels | 1 | 1 | 12 |
| top_channel_abs_mean_over_median_std | 16.64 | 18.31 | 74.26 |

### after_centering_then_layernorm

| quantity | `clip` | `perframe` | `levjepa` |
|---|---|---|---|
| mean_vector_norm | 0.3878 | 0.5122 | 1.311 |
| deviation_norm_mean | 32 | 32 | 31.97 |
| mean_to_deviation_ratio | 0.01212 | 0.01601 | 0.041 |
| energy_fraction_of_mean_direction | 0.0001469 | 0.0002562 | 0.001678 |
| pairwise_cosine_cross_sample | -0.004579 | -0.00436 | -0.004789 |
| pc1_variance_fraction_after_centering | 0.102 | 0.121 | 0.09279 |
| effective_rank_after_centering | 143.6 | 130.1 | 125.7 |
| massive_channels | 0 | 0 | 0 |
| top_channel_abs_mean_over_median_std | 0.1753 | 0.1824 | 0.3968 |

### scale (raw, L1)

| quantity | `clip` | `perframe` | `levjepa` |
|---|---|---|---|
| copy_last_l1 | 2.097 | 1.232 | 0.1553 |
| mean_abs_deviation_l1 | 1.72 | 1.683 | 0.2222 |
| mean_abs_value_l1 | 1.885 | 1.857 | 0.3737 |

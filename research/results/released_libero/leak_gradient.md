# Encoder leak and gradient decomposition: `C:/Users/vukst/vlajepa_local/models/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt`

64 samples, wm_loss_weight 0.1

## Encoder leak: relative change of each state when frames are perturbed

rel = ||s_k' - s_k|| / (per-state deviation scale); 0 = unchanged, 1 = as different as another sample.

### `clip` encoder

| perturbation | s_0 | s_1 | s_2 | s_3 |
|---|---|---|---|---|
| future_swap rel | 1.349 | 1.640 | 1.625 | 1.620 |
| future_swap cos | 0.6221 | 0.4550 | 0.4701 | 0.4438 |
| future_frozen rel | 1.353 | 1.413 | 1.487 | 1.560 |
| future_frozen cos | 0.6304 | 0.5979 | 0.5733 | 0.5064 |
| past_swap rel | 1.624 | 1.143 | 0.922 | 0.970 |
| past_swap cos | 0.4530 | 0.7351 | 0.8288 | 0.7991 |

### `perframe` encoder

| perturbation | s_0 | s_1 | s_2 | s_3 | s_4 | s_5 | s_6 | s_7 |
|---|---|---|---|---|---|---|---|---|
| future_swap rel | 0.000 | 0.000 | 1.639 | 1.639 | 1.639 | 1.639 | 1.638 | 1.638 |
| future_swap cos | 1.0000 | 1.0000 | 0.4687 | 0.4696 | 0.4691 | 0.4688 | 0.4717 | 0.4714 |
| future_frozen rel | 0.000 | 0.000 | 1.083 | 1.217 | 1.258 | 1.285 | 1.313 | 1.326 |
| future_frozen cos | 1.0000 | 1.0000 | 0.7586 | 0.6980 | 0.6772 | 0.6627 | 0.6496 | 0.6423 |
| past_swap rel | 1.641 | 1.641 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| past_swap cos | 0.4683 | 0.4676 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Gradient of the (weighted) world-model loss w.r.t. z

| quantity | mean | std |
|---|---|---|
| wm_loss | 1.13773 | 0.0226 |
| grad_norm_per_token | 7.57308e-08 | 9.38e-09 |
| grad_along_mean_dir_abs | 5.97956e-10 | 1.03e-10 |
| grad_off_mean_dir_norm | 7.57261e-08 | 9.38e-09 |
| sensitivity_shared_abs | 7.48018e-07 | 1.35e-07 |
| sensitivity_residual_abs | 1.5685e-07 | 2.15e-08 |
| z_shared_norm | 1163.83 | 46.2 |
| z_residual_norm | 282.088 | 40.7 |
| cos_grad_residual_abs | 0.0145698 | 0.00172 |
| cos_grad_mean_abs | 0.0115797 | 0.00273 |
| fd_shared_x1.1_minus_base | 5.52088e-07 | 7.62e-06 |
| fd_residual_x1.1_minus_base | 2.19569e-06 | 6.25e-06 |
| fd_residual_x2_minus_base | 1.30445e-05 | 3.58e-05 |
| fd_residual_x0_minus_base | 2.31713e-07 | 3.89e-06 |
| action_loss | 0.0341288 | 0.0234 |
| action_grad_norm_per_embodied_token | 0.000166903 | 0.000103 |
| wm_to_action_grad_ratio | 0.000638126 | 0.000381 |

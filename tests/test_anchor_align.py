import torch

from starVLA.model.modules.regularizers.anchor_align import (
    DIRECTION_WORDS,
    IGNORE_INDEX,
    build_anchor_mask,
    direction_labels,
    masked_layerwise_mse,
    pre_action_positions,
)


def test_build_anchor_mask_excludes_padding_and_special_tokens():
    ids = torch.tensor([[0, 5, 6, 900, 900, 901], [0, 0, 7, 8, 900, 901]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    keep = build_anchor_mask(ids, attn, excluded_token_ids=[900, 901])
    assert keep.tolist() == [[True, True, True, False, False, False], [False, False, True, True, False, False]]


def test_masked_layerwise_mse_zero_when_equal_and_ignores_masked():
    s = [torch.randn(2, 4, 8) for _ in range(3)]
    t = [x.clone() for x in s]
    keep = torch.ones(2, 4, dtype=torch.bool)
    loss, per_layer = masked_layerwise_mse(s, t, keep)
    assert loss.item() == 0.0 and per_layer.shape == (3,)
    # corrupt only masked-out positions -> still zero
    keep[:, 0] = False
    t2 = [x.clone() for x in s]
    for x in t2:
        x[:, 0] += 100.0
    loss2, _ = masked_layerwise_mse(s, t2, keep)
    assert loss2.item() == 0.0
    # corrupt a kept position -> positive, scaled by 0.5/sigma^2
    t3 = [x.clone() for x in s]
    t3[0][:, 1] += 1.0
    loss3, _ = masked_layerwise_mse(s, t3, keep, sigma=1.0)
    loss4, _ = masked_layerwise_mse(s, t3, keep, sigma=2.0)
    assert loss3 > 0 and torch.isclose(loss4, loss3 / 4)


def test_direction_labels_all_six_and_filter():
    K = 7
    chunk = torch.zeros(7, K, 7)
    chunk[0, :, 0] = 1.0   # +x forward
    chunk[1, :, 0] = -1.0  # -x backward
    chunk[2, :, 1] = 0.5   # +y left
    chunk[3, :, 1] = -0.5  # -y right
    chunk[4, :, 2] = 0.2   # +z up
    chunk[5, :, 2] = -0.2  # -z down
    chunk[6, :, :3] = 0.01  # near-stationary
    labels = direction_labels(chunk, min_norm=0.1)
    assert labels[:6].tolist() == [0, 1, 2, 3, 4, 5]
    assert labels[6].item() == IGNORE_INDEX
    assert [DIRECTION_WORDS[i] for i in labels[:6].tolist()] == ["forward", "backward", "left", "right", "up", "down"]


def test_direction_labels_zero_point_and_flip():
    chunk = torch.zeros(1, 4, 7)
    chunk[0, :, 2] = 0.1  # raw +z but zero point is 0.3 -> effectively -0.2 -> down
    assert direction_labels(chunk, zero_point=(0.0, 0.0, 0.3)).item() == 5
    assert direction_labels(chunk, zero_point=(0.0, 0.0, 0.3), flip_sign=(False, False, True)).item() == 4


def test_direction_labels_dominant_axis_uses_chunk_mean():
    chunk = torch.zeros(1, 2, 7)
    chunk[0, 0, 0] = 1.0
    chunk[0, 1, 0] = -1.0  # cancels -> x mean 0
    chunk[0, :, 1] = 0.3
    assert direction_labels(chunk).item() == 2  # left


def test_pre_action_positions():
    ids = torch.tensor([[1, 2, 3, 900, 900, 5], [900, 4, 4, 4, 4, 4], [1, 1, 1, 1, 1, 1]])
    pos = pre_action_positions(ids, action_token_ids=[900])
    assert pos.tolist() == [2, 0, 5]

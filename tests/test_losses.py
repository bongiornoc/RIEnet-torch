from __future__ import annotations

import numpy as np
import pytest
import torch

from rienet_torch.losses import variance_loss_function


def test_variance_loss_matches_manual_formula():
    covariance = torch.tensor(
        [
            [[2.0, 0.3], [0.3, 1.0]],
            [[1.5, 0.2], [0.2, 0.7]],
        ],
        dtype=torch.float32,
    )
    weights = torch.tensor(
        [
            [[0.25], [0.75]],
            [[0.6], [0.4]],
        ],
        dtype=torch.float32,
    )
    actual = variance_loss_function(covariance, weights)
    expected = []
    for cov, w in zip(covariance.numpy(), weights.numpy()):
        expected.append(2.0 * (w.T @ cov @ w))
    expected = np.stack(expected, axis=0)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("covariance", "weights"),
    [
        (torch.eye(3).repeat(2, 1, 1), torch.ones(1, 3, 1) / 3.0),
        (torch.eye(3), torch.ones(2, 3, 1) / 3.0),
        (torch.eye(3).repeat(2, 1, 1), torch.ones(2, 3) / 3.0),
        (torch.eye(3).repeat(2, 1, 1), torch.ones(2, 4, 1) / 4.0),
        (torch.eye(3).repeat(2, 1, 1), torch.ones(2, 3, 2) / 3.0),
        (torch.ones(2, 3, 4), torch.ones(2, 4, 1) / 4.0),
    ],
    ids=[
        "batch_mismatch",
        "rank_2_covariance",
        "rank_2_weights",
        "asset_mismatch",
        "weight_last_dim_not_one",
        "non_square_covariance",
    ],
)
def test_variance_loss_rejects_invalid_shapes(covariance, weights):
    with pytest.raises(ValueError):
        variance_loss_function(covariance, weights)


@pytest.mark.parametrize(
    ("covariance", "weights"),
    [
        (
            torch.tensor([[[1.0, 0.0], [0.0, torch.nan]]]),
            torch.ones(1, 2, 1) / 2.0,
        ),
        (
            torch.eye(2).unsqueeze(0),
            torch.tensor([[[0.5], [torch.inf]]]),
        ),
    ],
    ids=["nan_covariance", "inf_weights"],
)
def test_variance_loss_rejects_non_finite_inputs(covariance, weights):
    with pytest.raises(ValueError, match="finite"):
        variance_loss_function(covariance, weights)

from __future__ import annotations

import numpy as np
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

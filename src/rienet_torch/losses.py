"""
Loss functions module for RIEnet Torch.

This module provides the variance loss used to train RIEnet for global
minimum-variance (GMV) portfolio optimisation.

References
----------
Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
"Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage."
Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25).
Also see Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2026).
"End-to-End Large Portfolio Optimization for Variance Minimization with Neural
Networks through Covariance Cleaning" for a broader treatment.
"""

from __future__ import annotations

import torch

from .dtype_utils import ensure_float32


def variance_loss_function(
    covariance_true: torch.Tensor,
    weights_predicted: torch.Tensor,
) -> torch.Tensor:
    """
    Portfolio variance loss function for training RIEnet models.

    This loss function computes the global minimum-variance objective using the
    true covariance matrix and predicted portfolio weights.

    The portfolio variance is calculated as::

        variance = n_assets * w^T @ Sigma @ w

    where ``Sigma`` is the true covariance matrix and ``w`` are the predicted
    portfolio weights.

    Parameters
    ----------
    covariance_true : torch.Tensor
        Covariance matrices with shape ``(batch_size, n_assets, n_assets)``.
        Each matrix should be symmetric positive semi-definite.
    weights_predicted : torch.Tensor
        Predicted weights with shape ``(batch_size, n_assets, 1)``. The
        function assumes they already satisfy the portfolio constraint,
        typically sum-to-one across assets.

    Returns
    -------
    torch.Tensor
        Per-sample portfolio variance tensor with shape ``(batch_size, 1, 1)``.

    Notes
    -----
    The loss assumes:
    - daily returns data annualised upstream by the layer,
    - portfolio weights are expected to sum to one,
    - covariance matrices are positive semi-definite.

    Examples
    --------
    >>> import torch
    >>> from rienet_torch.losses import variance_loss_function
    >>> covariance = torch.randn(32, 10, 10)
    >>> covariance = covariance @ covariance.transpose(-1, -2)
    >>> weights = torch.randn(32, 10, 1)
    >>> weights = weights / weights.sum(dim=1, keepdim=True)
    >>> loss = variance_loss_function(covariance, weights)
    >>> print(loss.shape)
    torch.Size([32, 1, 1])
    """
    covariance_true = torch.as_tensor(covariance_true)
    weights_predicted = torch.as_tensor(weights_predicted)
    covariance_true, _ = ensure_float32(covariance_true)
    weights_predicted, _ = ensure_float32(weights_predicted)
    covariance_true = covariance_true.to(weights_predicted.dtype)
    n = torch.tensor(
        float(covariance_true.shape[-1]),
        dtype=weights_predicted.dtype,
        device=weights_predicted.device,
    )
    return n * weights_predicted.transpose(-1, -2) @ (covariance_true @ weights_predicted)

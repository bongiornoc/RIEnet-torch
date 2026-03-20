"""
Deterministic operation layers for RIEnet.

This module groups modules that do not own trainable parameters and only
perform tensor operations, statistics, matrix algebra, or normalization.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
from torch import nn

from .dtype_utils import ensure_float32, restore_dtype, epsilon_for_dtype

NormalizationModeType = Literal["inverse", "sum"]


class StandardDeviationLayer(nn.Module):
    """
    Module for computing sample standard deviation and mean.

    This module computes the sample standard deviation and mean along a
    specified axis, with optional demeaning for statistical preprocessing.

    Parameters
    ----------
    axis : int, default 1
        Axis along which to compute statistics.
    demean : bool, default False
        Whether to use an unbiased denominator ``n - 1``.
    epsilon : float, optional
        Small epsilon for numerical stability.
    name : str, optional
        Module name.
    """

    def __init__(
        self,
        axis: int = 1,
        demean: bool = False,
        epsilon: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the standard-deviation module.

        Parameters
        ----------
        axis : int, default 1
            Axis along which to compute statistics.
        demean : bool, default False
            Whether to use an unbiased denominator ``n - 1``.
        epsilon : float, optional
            Small epsilon for numerical stability.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("StandardDeviationLayer must have a name.")
        self.axis = axis
        self.demean = demean
        self.epsilon = float(epsilon if epsilon is not None else 1e-7)
        self.name = name

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-axis sample standard deviation and mean.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Statistics are computed along ``self.axis`` while
            preserving dimensions (``keepdim=True``), so the outputs can be
            broadcast back to ``x``.

        Returns
        -------
        tuple of torch.Tensor
            ``(std, mean)`` where both tensors have the same rank as ``x`` and
            singleton size on ``self.axis``.
        """
        x_work, original_dtype = ensure_float32(x)
        dtype = x_work.dtype
        epsilon = epsilon_for_dtype(dtype, self.epsilon).to(x_work.device)

        sample_size = torch.tensor(x_work.shape[self.axis], dtype=dtype, device=x_work.device)
        sample_size = torch.maximum(sample_size, torch.tensor(1.0, dtype=dtype, device=x_work.device))
        mean = x_work.mean(dim=self.axis, keepdim=True)
        centered = x_work - mean
        if self.demean:
            denom = torch.maximum(sample_size - 1.0, torch.tensor(1.0, dtype=dtype, device=x_work.device))
        else:
            denom = sample_size
        variance = centered.square().sum(dim=self.axis, keepdim=True) / denom
        std = torch.sqrt(torch.maximum(variance, epsilon))
        return restore_dtype(std, original_dtype), restore_dtype(mean, original_dtype)

    def get_config(self) -> dict:
        return {
            "axis": self.axis,
            "demean": self.demean,
            "epsilon": self.epsilon,
            "name": self.name,
        }


class CovarianceLayer(nn.Module):
    """
    Module for computing covariance matrices.

    This module computes sample covariance matrices from input data with
    optional normalization and dimension expansion.

    Parameters
    ----------
    expand_dims : bool, default False
        Whether to expand dimensions of the output.
    normalize : bool, default True
        Whether to normalize by sample size.
    name : str, optional
        Module name.
    """

    def __init__(
        self,
        expand_dims: bool = False,
        normalize: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize the covariance module.

        Parameters
        ----------
        expand_dims : bool, default False
            Whether to expand dimensions of the output.
        normalize : bool, default True
            Whether to normalize by sample size.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("CovarianceLayer must have a name.")
        self.expand_dims = expand_dims
        self.normalize = normalize
        self.name = name

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute batched covariance or correlation-like matrices.

        Parameters
        ----------
        returns : torch.Tensor
            Return tensor of shape ``(..., n_assets, n_observations)``. The
            last axis is interpreted as the sample axis.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., n_assets, n_assets)`` if
            ``expand_dims=False``; otherwise ``(..., 1, n_assets, n_assets)``.
        """
        returns_work, original_dtype = ensure_float32(returns)
        covariance = returns_work @ returns_work.transpose(-1, -2)
        if self.normalize:
            sample_size = torch.tensor(
                returns_work.shape[-1],
                dtype=returns_work.dtype,
                device=returns_work.device,
            )
            covariance = covariance / sample_size
        if self.expand_dims:
            covariance = covariance.unsqueeze(-3)
        return restore_dtype(covariance, original_dtype)

    def get_config(self) -> dict:
        return {
            "expand_dims": self.expand_dims,
            "normalize": self.normalize,
            "name": self.name,
        }


class SpectralDecompositionLayer(nn.Module):
    """
    Module for eigenvalue decomposition of symmetric matrices.

    This module performs eigenvalue decomposition using ``torch.linalg.eigh``,
    which is appropriate for symmetric or Hermitian matrices such as
    covariance matrices.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the spectral-decomposition module.

        Parameters
        ----------
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("SpectralDecompositionLayer must have a name.")
        self.name = name

    def forward(self, covariance_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform eigenvalue decomposition.

        Parameters
        ----------
        covariance_matrix : torch.Tensor
            Symmetric matrix tensor of shape ``(..., n, n)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(eigenvalues, eigenvectors)`` where:
            - ``eigenvalues`` has shape ``(..., n, 1)`` in ascending order.
            - ``eigenvectors`` has shape ``(..., n, n)``.
        """
        covariance32, original_dtype = ensure_float32(covariance_matrix)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance32)
        eigenvalues = eigenvalues.unsqueeze(-1)
        return restore_dtype(eigenvalues, original_dtype), restore_dtype(eigenvectors, original_dtype)

    def get_config(self) -> dict:
        return {"name": self.name}


class DimensionAwareLayer(nn.Module):
    """
    Module that builds per-asset dimensional attributes.

    This module returns only attribute channels, with shape
    ``(batch, n_stocks, k)`` where ``k == len(features)``.

    Parameters
    ----------
    features : list of str
        List of features to add: ``'n_stocks'``, ``'n_days'``, ``'q'``,
        ``'rsqrt_n_days'``.
    name : str, optional
        Module name.
    """

    def __init__(self, features: List[str], name: Optional[str] = None):
        """
        Initialize the dimensional-attribute module.

        Parameters
        ----------
        features : list of str
            List of features to add.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("DimensionAwareLayer must have a name.")
        self.features = list(features)
        self.name = name

    def _set_attribute(
        self,
        value: torch.Tensor,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        value = value.to(dtype=dtype, device=device).view(1, 1, 1)
        return value.expand(*shape)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Build dimensional attributes for each stock.

        Parameters
        ----------
        inputs : list of torch.Tensor
            Two-element list ``[standardized_returns, correlation_matrix]``:
            - ``standardized_returns``: shape ``(batch, n_stocks, n_days)``
            - ``correlation_matrix``: shape ``(batch, n_stocks, n_stocks)``
            The second tensor is used to infer the asset axis for broadcasting.

        Returns
        -------
        torch.Tensor
            Attribute tensor with shape ``(batch, n_stocks, k)``, where
            ``k == len(self.features)``.
        """
        standardized_returns, correlation_matrix = inputs
        n_stocks_raw = correlation_matrix.shape[-1]
        target_dtype = standardized_returns.dtype
        compute_dtype = torch.float32 if target_dtype in (torch.float16, torch.bfloat16) else target_dtype
        device = standardized_returns.device
        n_stocks = torch.tensor(float(n_stocks_raw), dtype=compute_dtype, device=device)
        n_days = torch.tensor(float(standardized_returns.shape[-1]), dtype=compute_dtype, device=device)
        final_shape = (correlation_matrix.shape[0], n_stocks_raw, 1)

        tensors_to_concat: List[torch.Tensor] = []
        if "q" in self.features:
            tensors_to_concat.append(self._set_attribute(n_days / n_stocks, final_shape, target_dtype, device))
        if "n_stocks" in self.features:
            tensors_to_concat.append(self._set_attribute(torch.sqrt(n_stocks), final_shape, target_dtype, device))
        if "n_days" in self.features:
            tensors_to_concat.append(self._set_attribute(torch.sqrt(n_days), final_shape, target_dtype, device))
        if "rsqrt_n_days" in self.features:
            tensors_to_concat.append(self._set_attribute(torch.rsqrt(n_days), final_shape, target_dtype, device))
        if not tensors_to_concat:
            return torch.zeros(
                (correlation_matrix.shape[0], n_stocks_raw, 0),
                dtype=target_dtype,
                device=device,
            )
        return torch.cat(tensors_to_concat, dim=-1)

    def get_config(self) -> dict:
        return {"features": self.features, "name": self.name}


class CustomNormalizationLayer(nn.Module):
    """
    Custom normalization module with different modes.

    This module applies different types of normalization along a specified
    axis, including sum normalization and inverse normalization.

    Parameters
    ----------
    mode : Literal['sum', 'inverse'], default 'sum'
        Normalization rule.
    axis : int, default -2
        Axis along which to normalize.
    inverse_power : float, default 1.0
        Exponent ``p`` used for inverse normalization. In
        ``mode='inverse'``, outputs are rescaled so that the mean of
        ``x^{-p}`` along ``axis`` is 1.
    epsilon : float, optional
        Small epsilon for numerical stability.
    name : str, optional
        Module name.
    """

    def __init__(
        self,
        mode: NormalizationModeType = "sum",
        axis: int = -2,
        inverse_power: float = 1.0,
        epsilon: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the normalization module.

        Parameters
        ----------
        mode : Literal['sum', 'inverse'], default 'sum'
            Normalization mode:
            - ``'sum'``: rescales values so the sum along ``axis`` equals the
              axis size.
            - ``'inverse'``: rescales values so the mean of ``x^{-p}`` along
              ``axis`` is 1.
        axis : int, default -2
            Axis used for normalization.
        inverse_power : float, default 1.0
            Inverse exponent ``p`` used only when ``mode='inverse'``.
        epsilon : float, optional
            Numerical epsilon for safe divisions and powers.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("CustomNormalizationLayer must have a name.")
        if inverse_power <= 0:
            raise ValueError("inverse_power must be positive")
        self.mode = mode
        self.axis = axis
        self.inverse_power = float(inverse_power)
        self.epsilon = float(epsilon if epsilon is not None else 1e-7)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization along the configured axis.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape as ``x``.

        Notes
        -----
        - ``mode='sum'`` enforces ``sum(x)=n`` along ``axis``.
        - ``mode='inverse'`` enforces ``mean(x^{-p})=1`` along ``axis``.
        """
        x_work, original_dtype = ensure_float32(x)
        epsilon = epsilon_for_dtype(x_work.dtype, self.epsilon).to(x_work.device)
        n = torch.tensor(float(x_work.shape[self.axis]), dtype=x_work.dtype, device=x_work.device)
        denom_axis = x_work.sum(dim=self.axis, keepdim=True)
        if self.mode == "sum":
            x_work = n * x_work / torch.maximum(denom_axis, epsilon)
        elif self.mode == "inverse":
            x_work = torch.maximum(x_work, epsilon)
            inv = torch.pow(x_work, -self.inverse_power)
            inv_total = inv.sum(dim=self.axis, keepdim=True)
            inv_normalized = n * inv / torch.maximum(inv_total, epsilon)
            x_work = torch.pow(torch.maximum(inv_normalized, epsilon), -1.0 / self.inverse_power)
        else:
            raise ValueError("Unsupported normalization mode")
        return restore_dtype(x_work, original_dtype)

    def get_config(self) -> dict:
        return {
            "mode": self.mode,
            "axis": self.axis,
            "inverse_power": self.inverse_power,
            "epsilon": self.epsilon,
            "name": self.name,
        }


class EigenvectorRescalingLayer(nn.Module):
    """
    Module that rescales eigenvectors to enforce unit diagonals.

    Given eigenvectors ``V`` and eigenvalues ``lambda`` this module computes the
    diagonal elements of ``V diag(lambda) V^T`` and divides each eigenvector
    row by the square root of the corresponding diagonal entry.

    Parameters
    ----------
    epsilon : float, optional
        Minimum value used to avoid division-by-zero during normalization.
    name : str, optional
        Module name.
    """

    def __init__(self, epsilon: Optional[float] = None, name: Optional[str] = None):
        """
        Initialize the eigenvector-rescaling module.

        Parameters
        ----------
        epsilon : float, optional
            Minimum value used to avoid division-by-zero during normalization.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("EigenvectorRescalingLayer must have a name.")
        self.epsilon = float(epsilon if epsilon is not None else 1e-7)
        self.name = name

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Rescale eigenvectors based on eigenvalues.

        Parameters
        ----------
        inputs : tuple
            Tuple ``(eigenvectors, eigenvalues)`` where:
            - ``eigenvectors`` has shape ``(..., n, n)``
            - ``eigenvalues`` has shape ``(..., n)`` or ``(..., n, 1)``

        Returns
        -------
        torch.Tensor
            Rescaled eigenvectors with the same shape as the input
            eigenvectors.
        """
        eigenvectors, eigenvalues = inputs
        eigenvectors_work, original_dtype = ensure_float32(torch.as_tensor(eigenvectors))
        dtype = eigenvectors_work.dtype
        eigenvalues = torch.as_tensor(eigenvalues, dtype=dtype, device=eigenvectors_work.device)
        eigenvalues = eigenvalues.reshape(eigenvectors_work.shape[:-1])
        diag = torch.einsum("...ij,...j,...ij->...i", eigenvectors_work, eigenvalues, eigenvectors_work)
        eps = epsilon_for_dtype(dtype, self.epsilon).to(eigenvectors_work.device)
        diag = torch.maximum(diag, eps)
        result = eigenvectors_work * torch.rsqrt(diag).unsqueeze(-1)
        return restore_dtype(result, original_dtype)

    def get_config(self) -> dict:
        return {"epsilon": self.epsilon, "name": self.name}


class EigenProductLayer(nn.Module):
    """
    Module for reconstructing matrices from eigenvalue decomposition.

    This module implements the vanilla reconstruction ``V diag(lambda) V^T``
    without any diagonal post-scaling. It assumes eigenvectors have already
    been preprocessed when diagonal control is required.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the eigen-product module.

        Parameters
        ----------
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("EigenProductLayer must have a name.")
        self.name = name

    def forward(self, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a matrix from an eigensystem.

        Parameters
        ----------
        eigenvalues : torch.Tensor
            Eigenvalues tensor with shape ``(..., n)`` or ``(..., n, 1)``.
        eigenvectors : torch.Tensor
            Eigenvectors tensor with shape ``(..., n, n)``.

        Returns
        -------
        torch.Tensor
            Reconstructed matrix ``V diag(lambda) V^T`` with shape
            ``(..., n, n)``.
        """
        eigenvectors_work, original_dtype = ensure_float32(torch.as_tensor(eigenvectors))
        dtype = eigenvectors_work.dtype
        eigenvalues = torch.as_tensor(eigenvalues, dtype=dtype, device=eigenvectors_work.device)
        eigenvalues = eigenvalues.reshape(eigenvectors_work.shape[:-1])
        scaled_vectors = eigenvectors_work * eigenvalues.unsqueeze(-2)
        result = scaled_vectors @ eigenvectors_work.transpose(-1, -2)
        return restore_dtype(result, original_dtype)

    def get_config(self) -> dict:
        return {"name": self.name}


class EigenWeightsLayer(nn.Module):
    """
    Compute GMV-like portfolio weights from eigensystem quantities.

    This module is intended for direct external use and accepts explicit
    inputs: eigenvectors, inverse eigenvalues, and optionally inverse standard
    deviations.

    Let ``V`` be eigenvectors and ``lambda_inv`` inverse eigenvalues. Define::

        c_k = sum_i V_{ik}
        s_k = lambda_inv_k * c_k

    The raw weights are computed as:
    - with ``inverse_std`` provided:
      ``raw_i = (sum_k V_{ik} s_k) * inverse_std_i``
    - with ``inverse_std=None``:
      ``raw_i = sum_k V_{ik} s_k``

    Then the output is normalized to sum to one across assets.

    Parameters
    ----------
    epsilon : float, optional
        Numerical stability term used for safe normalization.
    name : str, optional
        Module name.
    """

    def __init__(self, epsilon: Optional[float] = None, name: Optional[str] = None):
        """
        Initialize the eigen-weights module.

        Parameters
        ----------
        epsilon : float, optional
            Numerical stability term used for safe normalization.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("EigenWeightsLayer must have a name.")
        self.epsilon = float(epsilon if epsilon is not None else 1e-7)
        self.name = name

    def forward(
        self,
        eigenvectors: torch.Tensor,
        inverse_eigenvalues: torch.Tensor,
        inverse_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute normalized portfolio weights from spectral quantities.

        Parameters
        ----------
        eigenvectors : torch.Tensor
            Tensor of shape ``(..., n_assets, n_assets)``.
        inverse_eigenvalues : torch.Tensor
            Tensor of shape ``(..., n_assets)`` or ``(..., n_assets, 1)``.
        inverse_std : torch.Tensor, optional
            Optional tensor of shape ``(..., n_assets)`` or
            ``(..., n_assets, 1)`` representing inverse standard deviations.

        Returns
        -------
        torch.Tensor
            Normalized weights with shape ``(..., n_assets, 1)``.
        """
        eigenvectors_work, original_dtype = ensure_float32(torch.as_tensor(eigenvectors))
        dtype = eigenvectors_work.dtype
        device = eigenvectors_work.device

        inverse_eigenvalues = torch.as_tensor(inverse_eigenvalues, dtype=dtype, device=device)
        eigenvector_sum = eigenvectors_work.sum(dim=-2)
        target_shape = eigenvector_sum.shape
        inverse_eigenvalues = inverse_eigenvalues.reshape(target_shape)
        spectral_term = inverse_eigenvalues * eigenvector_sum
        raw_weights = (eigenvectors_work @ spectral_term.unsqueeze(-1)).squeeze(-1)
        if inverse_std is not None:
            inverse_std = torch.as_tensor(inverse_std, dtype=dtype, device=device).reshape(target_shape)
            raw_weights = raw_weights * inverse_std
        denom = raw_weights.sum(dim=-1, keepdim=True)
        epsilon = epsilon_for_dtype(dtype, self.epsilon).to(device)
        sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
        safe_denom = torch.where(denom.abs() < epsilon, sign * epsilon, denom)
        weights = raw_weights / safe_denom
        return restore_dtype(weights.unsqueeze(-1), original_dtype)

    def get_config(self) -> dict:
        return {"epsilon": self.epsilon, "name": self.name}


class NormalizedSum(nn.Module):
    """
    Sum first along ``axis_1`` and then normalize along ``axis_2``.

    This small helper is not part of the main paper-facing API, but it is kept
    public because it can be convenient when experimenting with custom RIEnet
    heads that require the same safe normalization rule.
    """

    def __init__(
        self,
        axis_1: int = -1,
        axis_2: int = -2,
        epsilon: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the normalized-sum module.

        Parameters
        ----------
        axis_1 : int, default -1
            First reduction axis.
        axis_2 : int, default -2
            Axis used for normalization after the first reduction.
        epsilon : float, optional
            Numerical stability term used for safe normalization.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("NormalizedSum must have a name.")
        self.axis_1 = axis_1
        self.axis_2 = axis_2
        self.epsilon = float(epsilon if epsilon is not None else 1e-7)
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the normalized sum.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor reduced along ``axis_1`` and normalized along ``axis_2``.
        """
        x_work, original_dtype = ensure_float32(x)
        epsilon = epsilon_for_dtype(x_work.dtype, self.epsilon).to(x_work.device)
        w = x_work.sum(dim=self.axis_1, keepdim=True)
        denominator = w.sum(dim=self.axis_2, keepdim=True)
        sign = torch.where(denominator >= 0, torch.ones_like(denominator), -torch.ones_like(denominator))
        safe_denominator = torch.where(denominator.abs() < epsilon, sign * epsilon, denominator)
        result = w / safe_denominator
        return restore_dtype(result, original_dtype)

    def get_config(self) -> dict:
        return {
            "axis_1": self.axis_1,
            "axis_2": self.axis_2,
            "epsilon": self.epsilon,
            "name": self.name,
        }

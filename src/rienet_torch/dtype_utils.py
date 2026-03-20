"""
Utility helpers to keep sensitive operations in float32 under mixed precision.

These functions centralise the logic for casting tensors to float32 when the
active dtype would otherwise request lower precision (for example float16 or
bfloat16) and for restoring the original dtype afterwards. They also provide
dtype-aware epsilon values so that stability constants track the active
precision.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

_LOWER_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def ensure_dense_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert sparse tensors to dense while preserving the logical value.

    RIEnet user-facing layers operate on dense tensors. When an input arrives
    in sparse form, this helper densifies it before further tensor operations.
    """
    if isinstance(tensor, torch.Tensor) and tensor.is_sparse:
        return tensor.to_dense()
    return torch.as_tensor(tensor)


def ensure_float32(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.dtype]]:
    """
    Cast ``tensor`` to float32 when it comes from a lower-precision dtype.

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.dtype]]
        The possibly cast tensor and the original dtype so callers can restore
        it after numerically sensitive computations.
    """
    dtype = getattr(tensor, "dtype", None)
    if dtype in _LOWER_PRECISION_DTYPES:
        return tensor.to(torch.float32), dtype
    return tensor, dtype


def restore_dtype(
    tensor: torch.Tensor,
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    Cast ``tensor`` back to ``dtype`` if it differs from the current dtype.
    """
    if dtype is None or tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


def epsilon_for_dtype(dtype: torch.dtype, base_value: float) -> torch.Tensor:
    """
    Return a stability epsilon scaled to ``dtype``.

    The helper guarantees the epsilon is at least as large as the machine
    epsilon of ``dtype`` while respecting the requested ``base_value``
    interpreted as the float32 baseline.
    """
    if dtype == torch.bfloat16:
        dtype_eps = 2.0 ** -7
    else:
        dtype_eps = float(np.finfo(torch.empty((), dtype=dtype).numpy().dtype).eps)
    scaled = max(float(base_value), float(dtype_eps))
    return torch.tensor(scaled, dtype=dtype)


def canonicalize_eigenvectors(eigenvectors: torch.Tensor) -> torch.Tensor:
    """
    Fix column signs deterministically so cross-framework comparisons are stable.

    Eigenvectors are defined only up to a sign flip. This helper chooses a
    deterministic sign for each column by forcing the entry with largest
    absolute value to be non-negative.
    """
    abs_vectors = eigenvectors.abs()
    pivot_index = abs_vectors.argmax(dim=-2, keepdim=True)
    pivot_values = torch.gather(eigenvectors, -2, pivot_index)
    signs = torch.where(
        pivot_values >= 0,
        torch.ones_like(pivot_values),
        -torch.ones_like(pivot_values),
    )
    return eigenvectors * signs

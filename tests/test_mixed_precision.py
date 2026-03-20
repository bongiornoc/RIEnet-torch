from __future__ import annotations

import pytest
import torch

from rienet_torch.ops_layers import StandardDeviationLayer
from rienet_torch.trainable_layers import LagTransformLayer


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_standard_deviation_restores_input_dtype(dtype: torch.dtype):
    x = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]], dtype=dtype)
    std, mean = StandardDeviationLayer(axis=-1, name=f"std_{dtype}")(x)
    assert std.dtype == dtype
    assert mean.dtype == dtype
    assert torch.isfinite(std).all()
    assert torch.isfinite(mean).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lag_transform_restores_input_dtype(dtype: torch.dtype):
    x = torch.linspace(-0.2, 0.2, steps=60, dtype=torch.float32).reshape(1, 3, 20).to(dtype=dtype)
    y = LagTransformLayer(variant="compact", name=f"lag_{dtype}")(x)
    assert y.dtype == dtype
    assert torch.isfinite(y).all()

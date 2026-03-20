from __future__ import annotations

import torch
from torch.nn import functional as F

from rienet_torch import RIEnetLayer


def _train_step(
    model: RIEnetLayer,
    inputs: torch.Tensor,
    target: torch.Tensor,
    *,
    lr: float = 1e-3,
) -> tuple[float, int]:
    outputs = model(inputs, training=True)
    loss = F.mse_loss(outputs, target)
    model.zero_grad(set_to_none=True)
    loss.backward()

    params_with_grad = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                params_with_grad += 1
                parameter.add_(parameter.grad, alpha=-lr)
    return float(loss.detach().cpu()), params_with_grad


def test_compact_variant_can_train_with_dynamic_time_dimension():
    torch.manual_seed(1234)
    model = RIEnetLayer(
        output_type="weights",
        lag_transform_variant="compact",
        name="dynamic_time_compact",
    )

    x_short = torch.randn(6, 5, 24)
    y_short = torch.randn(6, 5, 1)
    loss_short, grads_short = _train_step(model, x_short, y_short)

    x_long = torch.randn(6, 5, 33)
    y_long = torch.randn(6, 5, 1)
    loss_long, grads_long = _train_step(model, x_long, y_long)

    assert torch.isfinite(torch.tensor(loss_short))
    assert torch.isfinite(torch.tensor(loss_long))
    assert grads_short > 0
    assert grads_long > 0
    assert model(x_short).shape == (6, 5, 1)
    assert model(x_long).shape == (6, 5, 1)


def test_compact_variant_can_train_with_dynamic_stocks_and_time():
    torch.manual_seed(1234)
    model = RIEnetLayer(
        output_type="weights",
        lag_transform_variant="compact",
        name="dynamic_both_compact",
    )

    x_first = torch.randn(6, 4, 24)
    y_first = torch.randn(6, 4, 1)
    loss_first, grads_first = _train_step(model, x_first, y_first)

    x_second = torch.randn(6, 7, 31)
    y_second = torch.randn(6, 7, 1)
    loss_second, grads_second = _train_step(model, x_second, y_second)

    assert torch.isfinite(torch.tensor(loss_first))
    assert torch.isfinite(torch.tensor(loss_second))
    assert grads_first > 0
    assert grads_second > 0
    assert model(x_first).shape == (6, 4, 1)
    assert model(x_second).shape == (6, 7, 1)


def test_per_lag_variant_can_train_with_dynamic_stocks_and_fixed_time():
    torch.manual_seed(1234)
    n_days = 20
    model = RIEnetLayer(
        output_type="weights",
        lag_transform_variant="per_lag",
        name="dynamic_stocks_per_lag",
    )

    x_first = torch.randn(6, 5, n_days)
    y_first = torch.randn(6, 5, 1)
    loss_first, grads_first = _train_step(model, x_first, y_first)

    x_second = torch.randn(6, 9, n_days)
    y_second = torch.randn(6, 9, 1)
    loss_second, grads_second = _train_step(model, x_second, y_second)

    assert torch.isfinite(torch.tensor(loss_first))
    assert torch.isfinite(torch.tensor(loss_second))
    assert grads_first > 0
    assert grads_second > 0
    assert model(x_first).shape == (6, 5, 1)
    assert model(x_second).shape == (6, 9, 1)

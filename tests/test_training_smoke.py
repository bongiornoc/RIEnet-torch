from __future__ import annotations

import torch

from rienet_torch.losses import variance_loss_function
from rienet_torch.trainable_layers import RIEnetLayer


def test_rienet_training_step_stays_finite():
    torch.manual_seed(1234)
    model = RIEnetLayer(output_type="weights", name="train_smoke")

    returns = torch.randn(3, 5, 20)
    cov_seed = torch.randn(3, 5, 5)
    covariance = cov_seed @ cov_seed.transpose(-1, -2)

    weights = model(returns, training=True)
    loss = variance_loss_function(covariance, weights).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-1e-3)

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()

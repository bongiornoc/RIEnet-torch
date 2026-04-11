from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rienet_torch.losses import variance_loss_function
from rienet_torch.ops_layers import CustomNormalizationLayer, StandardDeviationLayer
from rienet_torch.rnn import KerasGRULayer
from rienet_torch.serialization import load_module, save_module
from rienet_torch.trainable_layers import CorrelationEigenTransformLayer, DeepLayer, RIEnetLayer


def test_sum_normalization_preserves_negative_denominator_sign():
    x = torch.tensor([[[-1.0], [-2.0], [0.5]]], dtype=torch.float32)
    y = CustomNormalizationLayer(mode="sum", axis=-2, name="negative_sum")(x)

    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.sum(dim=-2), torch.full((1, 1), 3.0))
    torch.testing.assert_close(y, 3.0 * x / x.sum(dim=-2, keepdim=True))


def test_sum_normalization_near_zero_denominator_stays_finite():
    x = torch.tensor([[[1.0], [-1.0 + 1e-9], [2e-9]]], dtype=torch.float32)
    y = CustomNormalizationLayer(mode="sum", axis=-2, epsilon=1e-6, name="near_zero_sum")(x)

    assert torch.isfinite(y).all()


def test_standard_deviation_demean_controls_unbiased_denominator_only():
    x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)

    std_biased, mean_biased = StandardDeviationLayer(axis=-1, demean=False, name="std_biased")(x)
    std_unbiased, mean_unbiased = StandardDeviationLayer(axis=-1, demean=True, name="std_unbiased")(x)

    torch.testing.assert_close(mean_biased, torch.tensor([[[2.0]]]))
    torch.testing.assert_close(mean_unbiased, torch.tensor([[[2.0]]]))
    torch.testing.assert_close(std_biased, torch.tensor([[[torch.sqrt(torch.tensor(2.0 / 3.0))]]]))
    torch.testing.assert_close(std_unbiased, torch.tensor([[[1.0]]]))


def test_rienet_single_asset_weights_are_exactly_one():
    torch.manual_seed(1234)
    model = RIEnetLayer(
        output_type="weights",
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="single_asset",
    )

    weights = model(torch.randn(3, 1, 6))

    torch.testing.assert_close(weights, torch.ones_like(weights), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("n_days", [1, 2])
def test_rienet_tiny_time_dimensions_stay_finite_and_normalized(n_days: int):
    torch.manual_seed(1234)
    model = RIEnetLayer(
        output_type="weights",
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name=f"tiny_time_{n_days}",
    )

    weights = model(torch.randn(2, 4, n_days))

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones((2, 1)), rtol=1e-6, atol=1e-6)


def test_rienet_weights_are_permutation_equivariant():
    torch.manual_seed(1234)
    model = RIEnetLayer(
        output_type="weights",
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="permutation_equivariant",
    )
    returns = torch.randn(2, 5, 8)
    permutation = torch.tensor([2, 4, 0, 3, 1])

    weights = model(returns)
    permuted_weights = model(returns[:, permutation, :])

    torch.testing.assert_close(permuted_weights, weights[:, permutation, :], rtol=2e-5, atol=2e-5)


def test_repeated_eigensystem_backward_keeps_rienet_gradients_finite():
    torch.manual_seed(1234)
    model = RIEnetLayer(
        output_type="weights",
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="zero_variance_grad",
    )
    returns = torch.zeros(2, 3, 4, requires_grad=True)
    covariance = torch.eye(3).repeat(2, 1, 1)

    weights = model(returns, training=True)
    loss = variance_loss_function(covariance, weights).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert returns.grad is not None
    assert torch.isfinite(returns.grad).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_variance_loss_backward_keeps_weight_gradients_finite():
    covariance = torch.tensor(
        [[[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]]],
        dtype=torch.float64,
    )
    weights = torch.tensor([[[0.2], [0.3], [0.5]]], dtype=torch.float64, requires_grad=True)

    loss = variance_loss_function(covariance, weights).mean()
    loss.backward()

    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


@pytest.mark.parametrize("bad_value", [torch.nan, torch.inf])
def test_rienet_rejects_nonfinite_inputs(bad_value: float):
    model = RIEnetLayer(
        output_type="weights",
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="nonfinite_input",
    )
    returns = torch.randn(1, 3, 4)
    returns[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        model(returns)


def test_correlation_transform_invalid_first_call_does_not_build_module():
    layer = CorrelationEigenTransformLayer(recurrent_layer_sizes=(2,), name="invalid_first_call")

    with pytest.raises(ValueError, match="Batch mismatch"):
        layer(torch.eye(3).unsqueeze(0).repeat(2, 1, 1), attributes=torch.randn(3, 2))

    assert layer.built is False


def test_train_mode_controls_rnn_dropout_when_training_argument_is_omitted():
    torch.manual_seed(1234)
    layer = KerasGRULayer(units=2, dropout=1.0, return_sequences=True, name="gru_dropout")
    inputs = torch.randn(2, 4, 3)

    layer.train()
    train_default = layer(inputs)
    train_explicit = layer(inputs, training=True)
    layer.eval()
    eval_default = layer(inputs)

    torch.testing.assert_close(train_default, train_explicit)
    assert not torch.allclose(train_default, eval_default)


def test_serialization_roundtrips_config_only_layers(tmp_path: Path):
    path = tmp_path / "std.pt"
    layer = StandardDeviationLayer(axis=-1, demean=True, epsilon=1e-4, name="std_roundtrip")

    save_module(layer, path)
    restored = load_module(StandardDeviationLayer, path)

    assert restored.get_config() == layer.get_config()
    std, mean = restored(torch.tensor([[[1.0, 2.0, 4.0]]]))
    assert torch.isfinite(std).all()
    assert torch.isfinite(mean).all()


def test_serialization_preserves_built_float64_module_dtype_and_outputs(tmp_path: Path):
    torch.manual_seed(1234)
    path = tmp_path / "deep64.pt"
    inputs = torch.randn(2, 3, dtype=torch.float64)
    layer = DeepLayer(hidden_layer_sizes=[4, 1], name="deep64")
    expected = layer(inputs)

    save_module(layer, path)
    restored = load_module(DeepLayer, path)
    actual = restored(inputs)

    assert all(parameter.dtype == torch.float64 for parameter in restored.parameters())
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_correlation_transform_rank_deficient_and_repeated_inputs_stay_finite():
    torch.manual_seed(1234)
    layer = CorrelationEigenTransformLayer(
        recurrent_layer_sizes=(2,),
        output_type=["correlation", "inverse_correlation"],
        name="rank_deficient",
    )
    repeated = torch.eye(4).unsqueeze(0)
    rank_deficient = torch.ones(1, 4, 4)
    nearly_singular = torch.full((1, 4, 4), 0.999999)
    nearly_singular.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    inputs = torch.cat([repeated, rank_deficient, nearly_singular], dim=0)

    outputs = layer(inputs)

    assert torch.isfinite(outputs["correlation"]).all()
    assert torch.isfinite(outputs["inverse_correlation"]).all()
    torch.testing.assert_close(
        outputs["correlation"],
        outputs["correlation"].transpose(-1, -2),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rienet_optimizer_steps_stay_finite_over_multiple_iterations():
    torch.manual_seed(1234)
    model = RIEnetLayer(
        output_type="weights",
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="multi_step_training",
    )
    returns = torch.randn(3, 4, 6)
    raw = torch.randn(3, 4, 4)
    covariance = raw @ raw.transpose(-1, -2)
    _ = model(returns, training=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(3):
        weights = model(returns, training=True)
        loss = variance_loss_function(covariance, weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss)
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        optimizer.step()
        for parameter in model.parameters():
            assert torch.isfinite(parameter).all()


def test_rienet_state_dict_roundtrip_output_parity(tmp_path: Path):
    torch.manual_seed(1234)
    path = tmp_path / "rienet_state.pt"
    inputs = torch.randn(2, 4, 6)
    model = RIEnetLayer(
        output_type=["weights", "precision"],
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="state_roundtrip",
    )
    expected = model(inputs)
    torch.save(model.state_dict(), path)

    restored = RIEnetLayer(
        output_type=["weights", "precision"],
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="state_roundtrip",
    )
    _ = restored(inputs)
    restored.load_state_dict(torch.load(path, map_location="cpu"))
    actual = restored(inputs)

    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-6)


def test_save_load_module_roundtrips_rienet_outputs_and_config(tmp_path: Path):
    torch.manual_seed(1234)
    path = tmp_path / "rienet_helper.pt"
    inputs = torch.randn(2, 4, 6)
    model = RIEnetLayer(
        output_type=["weights", "precision"],
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="helper_roundtrip",
    )
    expected = model(inputs)

    save_module(model, path)
    restored = load_module(RIEnetLayer, path)
    actual = restored(inputs)

    assert restored.get_config() == model.get_config()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-6)


def test_rienet_cuda_transfer_parity_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(1234)
    inputs = torch.randn(2, 4, 6)
    model = RIEnetLayer(
        output_type="weights",
        recurrent_layer_sizes=[2],
        std_hidden_layer_sizes=[2],
        name="cuda_parity",
    )
    expected = model(inputs)

    model = model.to("cuda")
    actual = model(inputs.to("cuda")).cpu()

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

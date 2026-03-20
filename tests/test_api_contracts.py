from __future__ import annotations

import numpy as np
import pytest
import torch

from rienet_torch import (
    CorrelationEigenTransformLayer as PublicCorrelationEigenTransformLayer,
    EigenWeightsLayer as PublicEigenWeightsLayer,
    LagTransformLayer as PublicLagTransformLayer,
    RIEnetLayer,
    variance_loss_function,
)
from rienet_torch.ops_layers import EigenProductLayer, EigenWeightsLayer
from rienet_torch.trainable_layers import CorrelationEigenTransformLayer, LagTransformLayer
from rienet_torch.rnn import KerasGRULayer, KerasLSTMLayer, KerasBidirectional


@pytest.mark.parametrize(
    ("output_type", "expected_shape"),
    [
        ("weights", (4, 6, 1)),
        ("precision", (4, 6, 6)),
        ("covariance", (4, 6, 6)),
        ("correlation", (4, 6, 6)),
        ("input_transformed", (4, 6, 20)),
        ("eigenvalues", (4, 6, 1)),
        ("eigenvectors", (4, 6, 6)),
        ("transformed_std", (4, 6, 1)),
    ],
)
def test_rienet_output_shapes(output_type: str, expected_shape: tuple[int, ...]):
    layer = RIEnetLayer(output_type=output_type, name=f"shape_{output_type}")
    outputs = layer(torch.randn(4, 6, 20))
    assert outputs.shape == expected_shape


def test_layer_initialization_defaults_and_invalid_output_type():
    layer = RIEnetLayer(name="defaults")
    assert layer.output_type == "weights"
    assert layer._direction == "bidirectional"
    assert layer._dimensional_features == ["n_stocks", "n_days", "q"]

    with pytest.raises(ValueError):
        RIEnetLayer(output_type="invalid", name="bad_output")


def test_public_exports_are_exposed_from_package_root():
    assert PublicEigenWeightsLayer is EigenWeightsLayer
    assert PublicCorrelationEigenTransformLayer is CorrelationEigenTransformLayer
    assert PublicLagTransformLayer is LagTransformLayer


def test_multiple_outputs_and_all_output_keys():
    multiple = RIEnetLayer(output_type=["weights", "precision"], name="multi")
    outputs = multiple(torch.randn(3, 4, 20))
    assert set(outputs.keys()) == {"weights", "precision"}
    assert outputs["weights"].shape == (3, 4, 1)
    assert outputs["precision"].shape == (3, 4, 4)

    everything = RIEnetLayer(output_type="all", name="all_outputs")
    outputs = everything(torch.randn(2, 4, 12))
    assert set(outputs.keys()) == {
        "weights",
        "precision",
        "covariance",
        "correlation",
        "input_transformed",
        "eigenvalues",
        "eigenvectors",
        "transformed_std",
    }


def test_weights_are_normalized_and_input_scaling_is_invariant():
    layer = RIEnetLayer(output_type="weights", name="scale_invariant")
    small_inputs = torch.ones((4, 5, 30)) * 0.001
    large_inputs = small_inputs * 252.0

    weights_small = layer(small_inputs)
    weights_large = layer(large_inputs)

    np.testing.assert_allclose(weights_small.sum(dim=1).detach().cpu().numpy(), 1.0, atol=1e-6)
    np.testing.assert_allclose(weights_small.detach().cpu().numpy(), weights_large.detach().cpu().numpy(), atol=1e-6)


def test_eigen_outputs_skip_std_transform_allocation():
    layer = RIEnetLayer(output_type=["eigenvalues", "eigenvectors"], name="eigen_only")
    outputs = layer(torch.randn(2, 5, 20))
    assert set(outputs.keys()) == {"eigenvalues", "eigenvectors"}
    assert layer.std_transform is None
    assert all("std_transform" not in name for name, _ in layer.named_parameters())


def test_additional_outputs_are_non_inverse_values():
    layer = RIEnetLayer(output_type=["eigenvalues", "transformed_std"], name="non_inverse")
    inputs = torch.randn(2, 4, 16)
    outputs = layer(inputs)

    scaled_inputs = inputs * layer._annualization_factor
    input_transformed = layer.lag_transform(scaled_inputs)
    std, mean = layer.std_layer(input_transformed)
    transformed_inverse_std = layer.std_transform(std)
    std_for_structural = transformed_inverse_std
    if layer.std_normalization is not None:
        std_for_structural = layer.std_normalization(transformed_inverse_std)

    zscores = (input_transformed - mean) / std
    correlation_matrix = layer.covariance_layer(zscores)
    attributes = layer.dimension_aware([zscores, correlation_matrix])
    transformed_inverse_eigenvalues = layer.correlation_eigen_transform(
        correlation_matrix,
        attributes=attributes,
        output_type="inverse_eigenvalues",
    )

    np.testing.assert_allclose(
        (outputs["eigenvalues"] * transformed_inverse_eigenvalues).detach().cpu().numpy(),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        (outputs["transformed_std"] * std_for_structural).detach().cpu().numpy(),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    )


def test_spectral_outputs_reconstruct_cleaned_correlation():
    layer = RIEnetLayer(output_type=["correlation", "eigenvalues", "eigenvectors"], name="spectral_reconstruct")
    outputs = layer(torch.randn(2, 5, 24))
    reconstructed = EigenProductLayer(name="product")(
        outputs["eigenvalues"].squeeze(-1),
        outputs["eigenvectors"],
    )
    np.testing.assert_allclose(
        reconstructed.detach().cpu().numpy(),
        outputs["correlation"].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_weights_match_cleaned_covariance_and_precision_solution():
    layer = RIEnetLayer(output_type=["weights", "covariance", "precision"], name="weight_solution")
    outputs = layer(torch.randn(3, 5, 24))

    covariance = outputs["covariance"].detach().cpu().numpy()
    precision = outputs["precision"].detach().cpu().numpy()
    weights = outputs["weights"].detach().cpu().numpy()
    ones = np.ones((weights.shape[0], weights.shape[1], 1), dtype=covariance.dtype)

    raw_from_covariance = np.linalg.solve(covariance, ones)
    expected_from_covariance = raw_from_covariance / raw_from_covariance.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(weights, expected_from_covariance, rtol=1e-5, atol=1e-6)

    raw_from_precision = precision @ ones
    expected_from_precision = raw_from_precision / raw_from_precision.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(weights, expected_from_precision, rtol=1e-5, atol=1e-6)


def test_custom_recurrent_configuration_and_direction_are_honored():
    layer = RIEnetLayer(
        output_type="weights",
        recurrent_layer_sizes=[12, 6],
        std_hidden_layer_sizes=[4, 2],
        recurrent_cell="LSTM",
        name="custom_recurrent",
    )
    weights = layer(torch.randn(2, 3, 15))
    assert weights.shape == (2, 3, 1)
    first_block = layer.correlation_eigen_transform.eigenvalue_transform.recurrent_layers[0]
    assert isinstance(first_block, KerasBidirectional)
    assert isinstance(first_block.forward_layer, KerasLSTMLayer)

    forward = RIEnetLayer(
        output_type="weights",
        recurrent_direction="forward",
        dimensional_features=["n_stocks", "rsqrt_n_days"],
        name="custom_direction",
    )
    weights = forward(torch.randn(2, 3, 15))
    assert weights.shape == (2, 3, 1)
    first_block = forward.correlation_eigen_transform.eigenvalue_transform.recurrent_layers[0]
    assert isinstance(first_block, KerasGRULayer)
    assert first_block.go_backwards is False
    assert forward.dimension_aware.features == ["n_stocks", "rsqrt_n_days"]


def test_invalid_recurrent_direction_and_dimensional_features_raise():
    with pytest.raises(ValueError, match="recurrent_direction"):
        RIEnetLayer(recurrent_direction="sideways", name="bad_direction")
    with pytest.raises(ValueError, match="dimensional_features"):
        RIEnetLayer(dimensional_features=["n_stocks", "bad_feature"], name="bad_features")


def test_covariance_diagonal_mean_is_centered_and_std_normalizer_can_be_disabled():
    covariance_layer = RIEnetLayer(output_type="covariance", name="cov_unit")
    covariance = covariance_layer(torch.randn(3, 4, 6))
    diag = torch.diagonal(covariance, dim1=-2, dim2=-1)
    mean_diag = diag.mean(dim=-1)
    np.testing.assert_allclose(mean_diag.detach().cpu().numpy(), 1.0, rtol=1e-5, atol=1e-4)

    no_norm = RIEnetLayer(
        output_type="precision",
        normalize_transformed_variance=False,
        name="no_std_norm",
    )
    assert no_norm.std_normalization is None


def test_variance_loss_shape_and_non_negative():
    batch_size, n_assets = 8, 5
    weights = torch.randn(batch_size, n_assets, 1)
    weights = weights / weights.sum(dim=1, keepdim=True)
    covariance = torch.eye(n_assets).unsqueeze(0).repeat(batch_size, 1, 1) * 0.01

    loss = variance_loss_function(covariance, weights)

    assert loss.shape == (batch_size, 1, 1)
    assert torch.all(loss >= 0)

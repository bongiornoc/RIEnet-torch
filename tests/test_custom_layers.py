from __future__ import annotations

import numpy as np
import pytest
import torch

from rienet_torch.ops_layers import (
    CovarianceLayer,
    DimensionAwareLayer,
    EigenProductLayer,
    EigenvectorRescalingLayer,
    EigenWeightsLayer,
    NormalizedSum,
    SpectralDecompositionLayer,
    StandardDeviationLayer,
)
from rienet_torch.trainable_layers import CorrelationEigenTransformLayer, DeepLayer, DeepRecurrentLayer, LagTransformLayer, RIEnetLayer


def test_standard_deviation_and_covariance_basic_contracts():
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    std, mean = StandardDeviationLayer(axis=-1, name="std_contract")(x)
    assert std.shape == (1, 2, 1)
    assert mean.shape == (1, 2, 1)
    assert torch.all(std > 0)

    returns = torch.randn(4, 3, 20)
    cov = CovarianceLayer(normalize=True, name="cov_contract")(returns)
    assert cov.shape == (4, 3, 3)
    np.testing.assert_allclose(cov.detach().cpu().numpy(), cov.transpose(-1, -2).detach().cpu().numpy(), rtol=1e-6)


def test_dimension_aware_q_is_n_stocks_over_n_days():
    batch_size, n_stocks, n_days = 2, 6, 50
    standardized_returns = torch.zeros(batch_size, n_stocks, n_days)
    correlation_matrix = torch.eye(n_stocks).repeat(batch_size, 1, 1)

    q = DimensionAwareLayer(features=["q"], name="q_ratio")(
        [standardized_returns, correlation_matrix]
    )

    expected = torch.full_like(q, n_stocks / n_days)
    torch.testing.assert_close(q, expected, rtol=1e-6, atol=1e-7)


def test_spectral_decomposition_and_deep_layers_basic_contracts():
    raw = torch.randn(4, 5, 5)
    cov = raw @ raw.transpose(-1, -2)
    eigenvalues, eigenvectors = SpectralDecompositionLayer(name="spectral_contract")(cov)
    assert eigenvalues.shape == (4, 5, 1)
    assert eigenvectors.shape == (4, 5, 5)
    assert torch.all(eigenvalues >= -1e-6)

    deep = DeepLayer(hidden_layer_sizes=[16, 8, 4], activation="relu", last_activation="linear", name="deep_contract")
    outputs = deep(torch.randn(8, 10))
    assert outputs.shape == (8, 4)

    recurrent = DeepRecurrentLayer(
        recurrent_layer_sizes=[32],
        recurrent_model="GRU",
        direction="bidirectional",
        name="deep_rnn_contract",
    )
    outputs = recurrent(torch.randn(4, 20, 8))
    assert outputs.shape[0] == 4


def test_normalized_sum_and_lag_transform_contracts():
    layer = NormalizedSum(axis_1=-1, axis_2=-2, name="norm_sum")
    weights = layer(torch.randn(8, 6, 6))
    np.testing.assert_allclose(weights.sum(dim=-2, keepdim=True).detach().cpu().numpy(), 1.0, rtol=1e-5)

    lag = LagTransformLayer(warm_start=True, name="lag_contract")
    returns = torch.randn(4, 5, 30) * 0.02
    transformed = lag(returns)
    assert transformed.shape == returns.shape
    assert not torch.allclose(transformed, returns)


def test_eigenvector_rescaling_and_eigen_weights_formulas():
    batch_size, n_assets = 3, 4
    eigenvalues = torch.rand(batch_size, n_assets) + 0.5
    eigenvectors = torch.linalg.qr(torch.randn(batch_size, n_assets, n_assets)).Q

    rescaled = EigenvectorRescalingLayer(name="rescale_contract")((eigenvectors, eigenvalues))
    reconstructed = EigenProductLayer(name="product_contract")(eigenvalues, rescaled)
    diag = torch.diagonal(reconstructed, dim1=-2, dim2=-1)
    assert float((diag - 1.0).abs().max()) < 1e-6

    weights_layer = EigenWeightsLayer(name="weights_contract")
    inverse_eigenvalues = torch.rand(batch_size, n_assets, 1) + 0.5
    weights_no_std = weights_layer(eigenvectors, inverse_eigenvalues)
    ev = eigenvectors.detach().cpu().numpy()
    inv_eig = inverse_eigenvalues.detach().cpu().numpy().reshape(batch_size, n_assets)
    c = ev.sum(axis=1)
    raw = np.einsum("bik,bk,bk->bi", ev, inv_eig, c)
    expected = raw / raw.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(weights_no_std.detach().cpu().numpy().squeeze(-1), expected, rtol=1e-5, atol=1e-6)


def test_eigen_weights_layer_matches_exact_gmv_from_covariance():
    covariance = torch.tensor(
        [
            [[2.0, 1.0, 0.0], [1.0, 2.0 / 3.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.5]],
        ],
        dtype=torch.float32,
    )
    std = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1))
    inverse_std = torch.reciprocal(std)
    correlation = covariance * inverse_std.unsqueeze(-1) * inverse_std.unsqueeze(-2)
    eigenvalues, eigenvectors = torch.linalg.eigh(correlation)

    weights = EigenWeightsLayer(name="exact_gmv")(
        eigenvectors,
        torch.reciprocal(eigenvalues),
        inverse_std.unsqueeze(-1),
    )

    ones = torch.ones_like(weights)
    raw_expected = torch.linalg.solve(covariance, ones)
    expected = raw_expected / raw_expected.sum(dim=-2, keepdim=True)
    torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        weights.sum(dim=-2),
        torch.ones_like(weights.sum(dim=-2)),
        rtol=1e-6,
        atol=1e-6,
    )
    assert float(weights[0, 0, 0]) < 0.0


def test_eigen_weights_layer_uses_inverse_std_on_both_sides():
    eigenvectors = torch.eye(3).unsqueeze(0)
    inverse_eigenvalues = torch.ones(1, 3)
    inverse_std = torch.tensor([[0.5, 1.0, 2.0]])

    weights = EigenWeightsLayer(name="two_sided_scale")(
        eigenvectors,
        inverse_eigenvalues,
        inverse_std,
    )

    raw_expected = inverse_std.square()
    expected = raw_expected / raw_expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        weights.squeeze(-1),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "row_scaled",
    [False, True],
    ids=["orthonormal", "positive_row_scaled"],
)
def test_eigen_weights_layer_matches_direct_inverse_correlation(row_scaled):
    torch.manual_seed(1234)
    batch_size, n_assets = 3, 5
    eigenvectors = torch.linalg.qr(
        torch.randn(batch_size, n_assets, n_assets, dtype=torch.float64)
    ).Q
    eigenvalues = torch.rand(batch_size, n_assets, dtype=torch.float64) * 1.5 + 0.5
    inverse_std = torch.rand(batch_size, n_assets, dtype=torch.float64) * 1.2 + 0.6

    if row_scaled:
        row_scale = torch.rand(batch_size, n_assets, dtype=torch.float64) * 1.5 + 0.5
        eigenvectors = eigenvectors * row_scale.unsqueeze(-1)

    weights = EigenWeightsLayer(name=f"direct_inverse_{row_scaled}")(
        eigenvectors,
        eigenvalues.reciprocal().unsqueeze(-1),
        inverse_std.unsqueeze(-1),
    )

    correlation = (
        eigenvectors
        @ torch.diag_embed(eigenvalues)
        @ eigenvectors.transpose(-1, -2)
    )
    raw_reference = inverse_std * torch.linalg.solve(
        correlation,
        inverse_std.unsqueeze(-1),
    ).squeeze(-1)
    reference = raw_reference / raw_reference.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        weights.squeeze(-1),
        reference,
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        weights.sum(dim=-2),
        torch.ones_like(weights.sum(dim=-2)),
        rtol=0.0,
        atol=1e-12,
    )


def test_correlation_eigen_transform_contracts_and_errors():
    layer = CorrelationEigenTransformLayer(name="corr_no_attr")
    raw = torch.randn(3, 5, 5)
    covariance = raw @ raw.transpose(-1, -2)
    std = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1))
    corr_scale = torch.einsum("bi,bj->bij", std, std)
    correlation = covariance / corr_scale
    cleaned = layer(correlation)
    assert cleaned.shape == (3, 5, 5)
    np.testing.assert_allclose(cleaned.detach().cpu().numpy(), cleaned.transpose(-1, -2).detach().cpu().numpy(), rtol=1e-5, atol=1e-6)

    layer = CorrelationEigenTransformLayer(name="corr_attr")
    correlation = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    attributes = torch.randn(2, 3)
    cleaned = layer(correlation, attributes=attributes)
    assert cleaned.shape == (2, 4, 4)

    layer = CorrelationEigenTransformLayer(name="corr_width")
    _ = layer(torch.eye(4).unsqueeze(0).repeat(2, 1, 1), attributes=torch.randn(2, 3))
    with pytest.raises(ValueError, match="Inconsistent eigenvalue feature width"):
        layer(torch.eye(4).unsqueeze(0).repeat(2, 1, 1))

    layer = CorrelationEigenTransformLayer(name="corr_batch")
    with pytest.raises(ValueError, match="Batch mismatch"):
        layer(torch.eye(4).unsqueeze(0).repeat(2, 1, 1), attributes=torch.randn(3, 2))

    layer = CorrelationEigenTransformLayer(output_type=["correlation", "inverse_correlation"], name="corr_inverse")
    raw = torch.randn(2, 4, 4)
    covariance = raw @ raw.transpose(-1, -2)
    std = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1))
    corr = covariance / torch.einsum("bi,bj->bij", std, std)
    outputs = layer(corr)
    identity = outputs["inverse_correlation"] @ outputs["correlation"]
    expected = np.broadcast_to(np.eye(4, dtype=np.float32), identity.shape)
    np.testing.assert_allclose(identity.detach().cpu().numpy(), expected, rtol=1e-5, atol=1e-5)

    inverse_only = CorrelationEigenTransformLayer(output_type="inverse_correlation", name="inverse_only")
    inverse = inverse_only(torch.eye(4).unsqueeze(0).repeat(2, 1, 1))
    assert inverse.shape == (2, 4, 4)
    np.testing.assert_allclose(inverse.detach().cpu().numpy(), inverse.transpose(-1, -2).detach().cpu().numpy(), rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="output_type"):
        CorrelationEigenTransformLayer(output_type="bad_output", name="bad_output")
    with pytest.raises(ValueError, match="output_type cannot be an empty sequence"):
        CorrelationEigenTransformLayer(output_type=[], name="empty_outputs")


def test_constructor_validation_contracts():
    with pytest.raises(ValueError, match="recurrent_model"):
        DeepRecurrentLayer(recurrent_layer_sizes=[4], recurrent_model="BAD", name="bad_model")
    with pytest.raises(ValueError, match="recurrent_cell"):
        CorrelationEigenTransformLayer(recurrent_cell="BAD", name="bad_cell")
    with pytest.raises(ValueError, match="output_type cannot be an empty sequence"):
        RIEnetLayer(output_type=[], name="empty_output")
    with pytest.raises(ValueError, match="recurrent_layer_sizes"):
        RIEnetLayer(recurrent_layer_sizes=[], name="empty_recurrent")
    with pytest.raises(ValueError, match="std_hidden_layer_sizes"):
        RIEnetLayer(std_hidden_layer_sizes=[], name="empty_std")

from __future__ import annotations

import numpy as np
import pytest
import torch

from rienet_torch.ops_layers import (
    CovarianceLayer,
    CustomNormalizationLayer,
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

    fixed_vectors = torch.tensor([[[1.0, 0.0, 0.0], [0.2, 0.9, 0.1], [0.3, 0.2, 0.8]]], dtype=torch.float32)
    fixed_inv = torch.tensor([[[1.2], [0.7], [1.1]]], dtype=torch.float32)
    fixed_std = torch.tensor([[[0.9], [1.0], [1.1]]], dtype=torch.float32)
    fixed_weights = weights_layer(fixed_vectors, fixed_inv, fixed_std)
    ev = fixed_vectors.numpy()
    inv_eig = fixed_inv.numpy().reshape(1, 3)
    inv_std = fixed_std.numpy().reshape(1, 3)
    c = ev.sum(axis=1)
    raw = np.einsum("bik,bk,bk,bi->bi", ev, inv_eig, c, inv_std)
    expected = raw / raw.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(fixed_weights.detach().cpu().numpy().squeeze(-1), expected, rtol=1e-6, atol=1e-7)


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

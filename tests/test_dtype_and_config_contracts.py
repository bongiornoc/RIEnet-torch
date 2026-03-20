from __future__ import annotations

import inspect

import pytest
import torch

from rienet_torch.losses import variance_loss_function
from rienet_torch.ops_layers import (
    CovarianceLayer,
    CustomNormalizationLayer,
    DimensionAwareLayer,
    EigenProductLayer,
    EigenvectorRescalingLayer,
    EigenWeightsLayer,
    NormalizedSum,
    SpectralDecompositionLayer,
    StandardDeviationLayer,
)
from rienet_torch.trainable_layers import (
    CorrelationEigenTransformLayer,
    DeepLayer,
    DeepRecurrentLayer,
    LagTransformLayer,
    RIEnetLayer,
)


GET_CONFIG_CASES = [
    (StandardDeviationLayer(axis=-2, demean=True, epsilon=1e-4, name="std_cfg"), {"axis": -2, "demean": True, "epsilon": 1e-4}),
    (CovarianceLayer(expand_dims=True, normalize=False, name="cov_cfg"), {"expand_dims": True, "normalize": False}),
    (SpectralDecompositionLayer(name="spectral_cfg"), {}),
    (DimensionAwareLayer(features=["n_days", "rsqrt_n_days"], name="dim_cfg"), {"features": ["n_days", "rsqrt_n_days"]}),
    (CustomNormalizationLayer(mode="inverse", axis=-1, inverse_power=2.5, epsilon=1e-4, name="norm_cfg"), {"mode": "inverse", "axis": -1, "inverse_power": 2.5, "epsilon": 1e-4}),
    (EigenvectorRescalingLayer(epsilon=1e-4, name="rescale_cfg"), {"epsilon": 1e-4}),
    (EigenProductLayer(name="product_cfg"), {}),
    (EigenWeightsLayer(epsilon=1e-4, name="weights_cfg"), {"epsilon": 1e-4}),
    (NormalizedSum(axis_1=-2, axis_2=-1, epsilon=1e-4, name="normsum_cfg"), {"axis_1": -2, "axis_2": -1, "epsilon": 1e-4}),
    (DeepLayer(hidden_layer_sizes=[5, 3], last_activation="softplus", activation="relu", other_biases=False, last_bias=False, dropout_rate=0.2, kernel_initializer="he_uniform", name="deep_cfg"), {"hidden_layer_sizes": [5, 3], "last_activation": "softplus", "activation": "relu", "other_biases": False, "last_bias": False, "dropout_rate": 0.2, "kernel_initializer": "he_uniform"}),
    (DeepRecurrentLayer(recurrent_layer_sizes=[7], final_activation="sigmoid", final_hidden_layer_sizes=[4], final_hidden_activation="relu", direction="forward", dropout=0.2, recurrent_dropout=0.1, recurrent_model="GRU", normalize="inverse", normalize_inverse_power=2.0, name="deeprnn_cfg"), {"recurrent_layer_sizes": [7], "final_activation": "sigmoid", "final_hidden_layer_sizes": [4], "final_hidden_activation": "relu", "direction": "forward", "dropout": 0.2, "recurrent_dropout": 0.1, "recurrent_model": "GRU", "normalize": "inverse", "normalize_inverse_power": 2.0}),
    (CorrelationEigenTransformLayer(recurrent_layer_sizes=(8,), recurrent_cell="LSTM", recurrent_direction="forward", final_hidden_layer_sizes=(4,), final_hidden_activation="relu", output_type=("correlation", "eigenvalues"), epsilon=1e-4, name="corr_cfg"), {"recurrent_layer_sizes": [8], "recurrent_cell": "LSTM", "recurrent_direction": "forward", "final_hidden_layer_sizes": [4], "final_hidden_activation": "relu", "output_type": ["correlation", "eigenvalues"], "epsilon": 1e-4}),
    (LagTransformLayer(warm_start=False, eps=1e-4, variant="per_lag", name="lag_cfg"), {"warm_start": False, "eps": 1e-4, "variant": "per_lag"}),
    (RIEnetLayer(output_type=["weights", "precision"], recurrent_layer_sizes=[8], std_hidden_layer_sizes=[4], recurrent_cell="LSTM", recurrent_direction="forward", dimensional_features=["n_days", "rsqrt_n_days"], normalize_transformed_variance=False, lag_transform_variant="per_lag", annualization_factor=365.0, name="rienet_cfg"), {"output_type": ["weights", "precision"], "recurrent_layer_sizes": [8], "std_hidden_layer_sizes": [4], "recurrent_cell": "LSTM", "recurrent_direction": "forward", "dimensional_features": ["n_days", "rsqrt_n_days"], "normalize_transformed_variance": False, "lag_transform_variant": "per_lag", "annualization_factor": 365.0}),
]


@pytest.mark.parametrize(("layer", "expected_config"), GET_CONFIG_CASES, ids=[layer.name for layer, _ in GET_CONFIG_CASES])
def test_get_config_includes_constructor_parameters(layer, expected_config):
    config = layer.get_config()
    init_signature = inspect.signature(layer.__class__.__init__)
    init_parameter_names = {
        name
        for name, parameter in init_signature.parameters.items()
        if name not in {"self", "kwargs"}
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    missing = init_parameter_names.difference(config.keys())
    assert missing == set()

    restored = layer.__class__.from_config(config) if hasattr(layer.__class__, "from_config") else layer.__class__(**config)
    restored_config = restored.get_config()
    for key, value in expected_config.items():
        assert restored_config[key] == value


def test_rienet_forward_bfloat16_stable():
    layer = RIEnetLayer(output_type="weights", name="rienet_bf16")
    x = torch.randn((2, 5, 20), dtype=torch.bfloat16)
    y = layer(x)
    assert y.dtype == torch.bfloat16
    assert bool(torch.isfinite(y.float()).all())


def test_layers_follow_float64_without_float32_hardcoding():
    lag_layer = LagTransformLayer(variant="compact", name="lag64")
    x = torch.randn((1, 4, 12), dtype=torch.float64)
    y = lag_layer(x)
    assert y.dtype == torch.float64
    assert all(weight.dtype == torch.float64 for weight in lag_layer.parameters())

    corr_layer = CorrelationEigenTransformLayer(output_type="correlation", name="corr64")
    corr = torch.eye(4, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    attrs = torch.ones((2, 4, 2), dtype=torch.float64)
    corr_out = corr_layer(corr, attributes=attrs)
    assert corr_out.dtype == torch.float64
    assert bool(torch.isfinite(corr_out).all())


def test_variance_loss_computes_in_float32_for_low_precision_inputs():
    covariance_true = torch.eye(4, dtype=torch.bfloat16).unsqueeze(0).repeat(2, 1, 1)
    weights = torch.ones((2, 4, 1), dtype=torch.bfloat16) / torch.tensor(4.0, dtype=torch.bfloat16)
    loss = variance_loss_function(covariance_true, weights)
    assert loss.dtype == torch.float32
    assert bool(torch.isfinite(loss).all())

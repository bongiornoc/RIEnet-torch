"""
Trainable layers for RIEnet.

This module groups all modules that include trainable parameters, including the
main end-to-end ``RIEnetLayer`` and reusable learnable subcomponents.
"""

from __future__ import annotations

import math
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from .dtype_utils import (
    ensure_dense_tensor,
    ensure_float32,
    restore_dtype,
    epsilon_for_dtype,
)
from .ops_layers import (
    NormalizationModeType,
    StandardDeviationLayer,
    CovarianceLayer,
    SpectralDecompositionLayer,
    DimensionAwareLayer,
    CustomNormalizationLayer,
    EigenvectorRescalingLayer,
    EigenProductLayer,
)
from .rnn import KerasDense, KerasGRULayer, KerasLSTMLayer, KerasBidirectional, resolve_training

LagTransformVariant = Literal["compact", "per_lag"]
RecurrentCellType = Literal["GRU", "LSTM"]
RecurrentDirectionType = Literal["bidirectional", "forward", "backward"]
CorrelationTransformOutput = Literal[
    "correlation",
    "inverse_correlation",
    "eigenvalues",
    "eigenvectors",
    "inverse_eigenvalues",
]
CorrelationTransformOutputType = Union[
    CorrelationTransformOutput,
    Literal["all"],
    Sequence[Union[CorrelationTransformOutput, Literal["all"]]],
]

OutputComponent = Literal[
    "weights",
    "precision",
    "covariance",
    "correlation",
    "input_transformed",
    "eigenvalues",
    "eigenvectors",
    "transformed_std",
]
OutputToken = Union[OutputComponent, Literal["all"]]
OutputType = Union[OutputToken, Sequence[OutputToken]]
RecurrentCell = RecurrentCellType
RecurrentDirection = RecurrentDirectionType
DimensionalFeature = Literal["n_stocks", "n_days", "q", "rsqrt_n_days"]


class DeepLayer(nn.Module):
    """
    Multi-layer dense network with configurable activation and dropout.

    This module implements a sequence of dense layers with specified
    activations, dropout, and flexible configuration for the final layer.

    Parameters
    ----------
    hidden_layer_sizes : list of int
        Sizes of hidden layers including the output layer.
    last_activation : str, default "linear"
        Activation for the final layer.
    activation : str, default "leaky_relu"
        Activation for hidden layers.
    other_biases : bool, default True
        Whether to use bias in hidden layers.
    last_bias : bool, default True
        Whether to use bias in the final layer.
    dropout_rate : float, default 0.0
        Dropout rate for hidden layers.
    kernel_initializer : str, default "glorot_uniform"
        Weight initialization method.
    name : str, optional
        Module name.
    """

    def __init__(
        self,
        hidden_layer_sizes: List[int],
        last_activation: str = "linear",
        activation: str = "leaky_relu",
        other_biases: bool = True,
        last_bias: bool = True,
        dropout_rate: float = 0.0,
        kernel_initializer: str = "glorot_uniform",
        name: Optional[str] = None,
    ):
        """
        Initialize the dense stack.

        Parameters
        ----------
        hidden_layer_sizes : list of int
            Sizes of hidden layers including the output layer.
        last_activation : str, default "linear"
            Activation for the final layer.
        activation : str, default "leaky_relu"
            Activation for hidden layers.
        other_biases : bool, default True
            Whether to use bias in hidden layers.
        last_bias : bool, default True
            Whether to use bias in the final layer.
        dropout_rate : float, default 0.0
            Dropout rate for hidden layers.
        kernel_initializer : str, default "glorot_uniform"
            Weight initialization method.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("DeepLayer must have a name.")
        if not hidden_layer_sizes:
            raise ValueError("hidden_layer_sizes must contain at least one positive integer.")
        if any(size <= 0 for size in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive integers.")
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.activation = activation
        self.last_activation = last_activation
        self.other_biases = other_biases
        self.last_bias = last_bias
        self.dropout_rate = float(dropout_rate)
        self.kernel_initializer = kernel_initializer
        self.name = name

        self.hidden_layers = nn.ModuleList(
            [
                KerasDense(
                    size,
                    activation=self.activation,
                    use_bias=self.other_biases,
                    kernel_initializer=self.kernel_initializer,
                    name=f"{self.name}_hidden_{i}",
                )
                for i, size in enumerate(self.hidden_layer_sizes[:-1])
            ]
        )
        self.final_dense = KerasDense(
            self.hidden_layer_sizes[-1],
            activation=self.last_activation,
            use_bias=self.last_bias,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_output",
        )
        self.built = False

    def build(self, input_shape) -> None:
        current_shape = tuple(input_shape)
        self._build_spec = (tuple(input_shape),)
        for dense in self.hidden_layers:
            dense.build(current_shape)
            current_shape = (*current_shape[:-1], dense.units)
        self.final_dense.build(current_shape)
        self.built = True

    def forward(self, inputs: torch.Tensor, training: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape ``(..., features)``.
        training : bool, optional
            Training flag controlling dropout behavior.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., hidden_layer_sizes[-1])``.
        """
        if not self.built:
            self.build(inputs.shape)
            self.to(device=inputs.device, dtype=inputs.dtype)
        is_training = resolve_training(self, training)
        x = inputs
        for dense in self.hidden_layers:
            x = dense(x)
            x = F.dropout(x, p=self.dropout_rate, training=is_training and self.dropout_rate > 0.0)
        return self.final_dense(x)

    def get_config(self) -> dict:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "last_activation": self.last_activation,
            "other_biases": self.other_biases,
            "last_bias": self.last_bias,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": self.kernel_initializer,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


class DeepRecurrentLayer(nn.Module):
    """
    Deep recurrent module with configurable RNN cells and post-processing.

    This module implements a stack of recurrent layers (LSTM or GRU) with
    optional bidirectional processing, followed by dense layers for final
    transformation.

    Parameters
    ----------
    recurrent_layer_sizes : list of int
        Sizes of recurrent layers.
    final_activation : str, default "softplus"
        Activation for the final dense layer.
    final_hidden_layer_sizes : list of int, default []
        Hidden sizes of the post-RNN MLP head before the final 1-unit output.
    final_hidden_activation : str, default "leaky_relu"
        Activation for final hidden layers.
    direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
        RNN direction strategy.
    dropout : float, default 0.0
        Dropout rate for RNN layers and hidden dense layers.
    recurrent_dropout : float, default 0.0
        Recurrent dropout rate.
    recurrent_model : Literal['LSTM', 'GRU'], default 'LSTM'
        Type of recurrent cell.
    normalize : Literal['inverse', 'sum'] or None, optional
        Post-projection normalization mode applied along the sequence axis.
    normalize_inverse_power : float, default 1.0
        Exponent ``p`` used only when ``normalize='inverse'``.
    name : str, optional
        Module name.
    """

    def __init__(
        self,
        recurrent_layer_sizes: List[int],
        final_activation: str = "softplus",
        final_hidden_layer_sizes: List[int] = [],
        final_hidden_activation: str = "leaky_relu",
        direction: RecurrentDirectionType = "bidirectional",
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        recurrent_model: RecurrentCellType = "LSTM",
        normalize: Optional[NormalizationModeType] = None,
        normalize_inverse_power: float = 1.0,
        name: Optional[str] = None,
    ):
        """
        Initialize a stacked recurrent block followed by an MLP head.

        Parameters
        ----------
        recurrent_layer_sizes : list of int
            Number of units for each recurrent layer.
        final_activation : str, default "softplus"
            Activation used by the final 1-unit output layer.
        final_hidden_layer_sizes : list of int, default []
            Hidden sizes for the post-RNN MLP head.
        final_hidden_activation : str, default "leaky_relu"
            Activation used by hidden layers in the MLP head.
        direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
            RNN direction mode:
            - ``'bidirectional'``: process sequence in both directions.
            - ``'forward'``: process left-to-right only.
            - ``'backward'``: process right-to-left only.
        dropout : float, default 0.0
            Input dropout used in recurrent layers and MLP hidden layers.
        recurrent_dropout : float, default 0.0
            Recurrent-state dropout used inside each recurrent cell.
        recurrent_model : Literal['LSTM', 'GRU'], default 'LSTM'
            Recurrent cell type.
        normalize : Literal['inverse', 'sum'] or None, optional
            Optional output normalization applied along the time axis.
        normalize_inverse_power : float, default 1.0
            Inverse normalization exponent, used only when
            ``normalize='inverse'``.
        name : str, optional
            Module name.
        """
        super().__init__()
        if name is None:
            raise ValueError("DeepRecurrentLayer must have a name.")
        if not recurrent_layer_sizes:
            raise ValueError("recurrent_layer_sizes must contain at least one positive integer.")
        if any(size <= 0 for size in recurrent_layer_sizes):
            raise ValueError("recurrent_layer_sizes must contain positive integers.")
        if any(size <= 0 for size in final_hidden_layer_sizes):
            raise ValueError("final_hidden_layer_sizes must contain positive integers.")
        if normalize not in [None, "inverse", "sum"]:
            raise ValueError("normalize must be None, 'inverse', or 'sum'.")
        if normalize is not None and normalize_inverse_power <= 0:
            raise ValueError("normalize_inverse_power must be positive when using inverse normalization.")
        if recurrent_model not in {"GRU", "LSTM"}:
            raise ValueError("recurrent_model must be either 'GRU' or 'LSTM'.")

        self.recurrent_layer_sizes = list(recurrent_layer_sizes)
        self.final_activation = final_activation
        self.final_hidden_layer_sizes = list(final_hidden_layer_sizes)
        self.final_hidden_activation = final_hidden_activation
        self.direction = direction
        self.dropout = float(dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.recurrent_model = recurrent_model
        self.normalize = normalize
        self.normalize_inverse_power = float(normalize_inverse_power)
        self.name = name
        self.built = False

        cell_cls = KerasGRULayer if recurrent_model == "GRU" else KerasLSTMLayer
        layers_: List[nn.Module] = []
        for i, units in enumerate(self.recurrent_layer_sizes):
            layer_name = f"{self.name}_rnn_{i}"
            if direction == "bidirectional":
                cell = cell_cls(
                    units=units,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True,
                    go_backwards=False,
                    name=f"{layer_name}_cell",
                )
                rnn_layer = KerasBidirectional(cell, name=layer_name)
            elif direction == "forward":
                rnn_layer = cell_cls(
                    units=units,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True,
                    go_backwards=False,
                    name=layer_name,
                )
            elif direction == "backward":
                rnn_layer = cell_cls(
                    units=units,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True,
                    go_backwards=True,
                    name=layer_name,
                )
            else:
                raise ValueError("direction must be 'bidirectional', 'forward', or 'backward'.")
            layers_.append(rnn_layer)
        self.recurrent_layers = nn.ModuleList(layers_)
        self.final_deep_dense = DeepLayer(
            self.final_hidden_layer_sizes + [1],
            activation=self.final_hidden_activation,
            last_activation=self.final_activation,
            dropout_rate=self.dropout,
            name=f"{self.name}_finaldeep",
        )
        self._normalizer = None
        if self.normalize is not None:
            inverse_power = self.normalize_inverse_power if self.normalize == "inverse" else 1.0
            self._normalizer = CustomNormalizationLayer(
                mode=self.normalize,
                axis=-2,
                inverse_power=inverse_power,
                name=f"{self.name}_norm",
            )

    def build(self, input_shape) -> None:
        current_shape = tuple(input_shape)
        self._build_spec = (tuple(input_shape),)
        for layer in self.recurrent_layers:
            if hasattr(layer, "build"):
                layer.build(current_shape)
            feature_width = layer.forward_layer.units * 2 if isinstance(layer, KerasBidirectional) else layer.units
            current_shape = (*current_shape[:-1], feature_width)
        self.final_deep_dense.build(current_shape)
        self.built = True

    def forward(self, inputs: torch.Tensor, training: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass through the recurrent stack and projection head.

        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence tensor with shape ``(batch, timesteps, features)``.
        training : bool, optional
            Training flag controlling recurrent and dense dropout.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(batch, timesteps)`` after squeezing the final
            singleton feature axis.
        """
        if not self.built:
            self.build(inputs.shape)
            self.to(device=inputs.device, dtype=inputs.dtype)
        is_training = resolve_training(self, training)
        x = inputs
        for layer in self.recurrent_layers:
            x = layer(x, training=is_training)
        outputs = self.final_deep_dense(x, training=is_training)
        if self._normalizer is not None:
            outputs = self._normalizer(outputs)
        return outputs.squeeze(-1)

    def get_config(self) -> dict:
        return {
            "recurrent_layer_sizes": self.recurrent_layer_sizes,
            "final_activation": self.final_activation,
            "final_hidden_layer_sizes": self.final_hidden_layer_sizes,
            "final_hidden_activation": self.final_hidden_activation,
            "direction": self.direction,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "recurrent_model": self.recurrent_model,
            "normalize": self.normalize,
            "normalize_inverse_power": self.normalize_inverse_power,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


class CorrelationEigenTransformLayer(nn.Module):
    """
    Transform a correlation matrix by cleaning its eigenvalues.

    The module performs:
    1. Eigen-decomposition of the input correlation matrix.
    2. Optional enrichment of each eigenvalue with per-batch or per-asset
       attributes.
    3. Recurrent transformation of enriched eigenvalue features in
       inverse-eigenvalue space.
    4. Reconstruction of a cleaned correlation matrix with diagonal rescaling.

    Parameters
    ----------
    recurrent_layer_sizes : tuple of int, default (16,)
        Hidden sizes of recurrent layers used to transform eigenvalues.
    recurrent_cell : Literal['GRU', 'LSTM'], default 'GRU'
        Recurrent cell family.
    recurrent_direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
        Direction for recurrent processing.
    final_hidden_layer_sizes : tuple of int, default ()
        Hidden sizes of the post-recurrent MLP head.
    final_hidden_activation : str, default 'leaky_relu'
        Activation for optional dense hidden layers.
    output_type : CorrelationTransformOutputType, default 'correlation'
        Requested output component(s). Allowed values are ``'correlation'``,
        ``'inverse_correlation'``, ``'eigenvalues'``, ``'eigenvectors'``,
        ``'inverse_eigenvalues'`` or ``'all'``.
    epsilon : float, optional
        Numerical epsilon used before reciprocal operations.
    name : str, optional
        Module name.
    """

    _ALLOWED_OUTPUTS = (
        "correlation",
        "inverse_correlation",
        "eigenvalues",
        "eigenvectors",
        "inverse_eigenvalues",
    )

    def __init__(
        self,
        recurrent_layer_sizes: Tuple[int, ...] = (16,),
        recurrent_cell: RecurrentCellType = "GRU",
        recurrent_direction: RecurrentDirectionType = "bidirectional",
        final_hidden_layer_sizes: Tuple[int, ...] = (),
        final_hidden_activation: str = "leaky_relu",
        output_type: CorrelationTransformOutputType = "correlation",
        epsilon: Optional[float] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the correlation-eigenvalue cleaning module.

        Parameters
        ----------
        recurrent_layer_sizes : tuple of int, default (16,)
            Units for each recurrent layer in the eigenvalue cleaning block.
        recurrent_cell : Literal['GRU', 'LSTM'], default 'GRU'
            Recurrent cell type used in the cleaning block.
        recurrent_direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
            Sequence direction mode.
        final_hidden_layer_sizes : tuple of int, default ()
            Hidden sizes of the post-recurrent MLP head.
        final_hidden_activation : str, default 'leaky_relu'
            Activation for hidden layers in the post-recurrent MLP head.
        output_type : CorrelationTransformOutputType, default 'correlation'
            Requested output component(s). A sequence can be passed to request
            multiple components.
        epsilon : float, optional
            Numerical epsilon used in safe reciprocal operations.
        name : str, optional
            Module name.
        """
        super().__init__()
        recurrent_layer_sizes = list(recurrent_layer_sizes)
        final_hidden_layer_sizes = list(final_hidden_layer_sizes)
        if not recurrent_layer_sizes:
            raise ValueError("recurrent_layer_sizes must contain at least one positive integer.")
        if any(units <= 0 for units in recurrent_layer_sizes):
            raise ValueError("recurrent_layer_sizes must contain positive integers.")
        if any(units <= 0 for units in final_hidden_layer_sizes):
            raise ValueError("final_hidden_layer_sizes must contain positive integers.")
        self._output_config = output_type if isinstance(output_type, str) else list(output_type)
        self.output_components = tuple(self._resolve_output_components(output_type))
        self.output_type = self.output_components[0] if len(self.output_components) == 1 else tuple(self.output_components)
        self.recurrent_layer_sizes = recurrent_layer_sizes
        self.recurrent_cell = recurrent_cell.strip().upper()
        self.recurrent_direction = recurrent_direction.strip().lower()
        self.final_hidden_layer_sizes = final_hidden_layer_sizes
        self.final_hidden_activation = final_hidden_activation
        self.epsilon = float(epsilon if epsilon is not None else 1e-7)
        self._feature_width: Optional[int] = None
        self.name = name
        self.built = False
        if self.recurrent_cell not in {"GRU", "LSTM"}:
            raise ValueError("recurrent_cell must be either 'GRU' or 'LSTM'.")
        if self.recurrent_direction not in {"bidirectional", "forward", "backward"}:
            raise ValueError("recurrent_direction must be 'bidirectional', 'forward', or 'backward'.")

        self.spectral_decomp = SpectralDecompositionLayer(name=f"{self.name}_spectral")
        self.eigenvalue_transform = DeepRecurrentLayer(
            recurrent_layer_sizes=self.recurrent_layer_sizes,
            recurrent_model=self.recurrent_cell,
            direction=self.recurrent_direction,
            dropout=0.0,
            recurrent_dropout=0.0,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            final_hidden_activation=self.final_hidden_activation,
            final_activation="softplus",
            normalize="inverse",
            normalize_inverse_power=1.0,
            name=f"{self.name}_eigenvalue_rnn",
        )
        self.eigenvector_rescaler = EigenvectorRescalingLayer(
            epsilon=self.epsilon,
            name=f"{self.name}_eigenvector_rescaler",
        )
        self.correlation_product = EigenProductLayer(name=f"{self.name}_correlation")

    def _resolve_output_components(self, output_type: CorrelationTransformOutputType) -> List[str]:
        if isinstance(output_type, str):
            if output_type == "all":
                return list(self._ALLOWED_OUTPUTS)
            if output_type not in self._ALLOWED_OUTPUTS:
                raise ValueError("output_type must be a valid correlation output.")
            return [output_type]
        if not output_type:
            raise ValueError("output_type cannot be an empty sequence.")
        expanded: List[str] = []
        for entry in output_type:
            if entry == "all":
                expanded.extend(self._ALLOWED_OUTPUTS)
            elif entry in self._ALLOWED_OUTPUTS:
                expanded.append(entry)
            else:
                raise ValueError("All requested outputs must be valid correlation outputs.")
        deduped: List[str] = []
        for entry in expanded:
            if entry not in deduped:
                deduped.append(entry)
        return deduped

    def build(self, correlation_matrix_shape, attributes_shape=None) -> None:
        """
        Build internal submodules with explicit shapes.

        Parameters
        ----------
        correlation_matrix_shape : tuple
            Correlation matrix shape ``(batch, n_assets, n_assets)``.
        attributes_shape : tuple, optional
            Optional attributes shape ``(batch, k)`` or ``(batch, n_assets, k)``.
        """
        corr_shape = tuple(correlation_matrix_shape)
        attrs_shape = None if attributes_shape is None else tuple(attributes_shape)
        self._build_spec = (corr_shape, attrs_shape)
        attr_width = 0
        if attributes_shape is not None:
            attr_width = int(attributes_shape[-1])
        feature_width = 1 + attr_width
        self._feature_width = feature_width
        self.spectral_decomp
        self.eigenvalue_transform.build((corr_shape[0], corr_shape[-1], feature_width))
        self.built = True

    def _exact_inverse_from_rescaled_eigensystem(
        self,
        direct_eigenvectors: torch.Tensor,
        inverse_eigenvalues: torch.Tensor,
    ) -> torch.Tensor:
        vectors_work, original_dtype = ensure_float32(torch.as_tensor(direct_eigenvectors))
        dtype = vectors_work.dtype
        inverse_eigenvalues = torch.as_tensor(inverse_eigenvalues, dtype=dtype, device=vectors_work.device)
        inverse_eigenvalues = inverse_eigenvalues.reshape(vectors_work.shape[:-1])
        row_norm_sq = vectors_work.square().sum(dim=-1)
        eps = epsilon_for_dtype(dtype, self.epsilon).to(vectors_work.device)
        inverse_row_norm_sq = torch.reciprocal(torch.maximum(row_norm_sq, eps))
        scaled_vectors = vectors_work * inverse_row_norm_sq.unsqueeze(-1)
        inverse_work = (scaled_vectors * inverse_eigenvalues.unsqueeze(-2)) @ scaled_vectors.transpose(-1, -2)
        inverse_work = 0.5 * (inverse_work + inverse_work.transpose(-1, -2))
        return restore_dtype(inverse_work, original_dtype)

    def forward(
        self,
        correlation_matrix: torch.Tensor,
        attributes: Optional[torch.Tensor] = None,
        output_type: Optional[CorrelationTransformOutputType] = None,
        include_raw_eigenvectors: bool = False,
        training: Optional[bool] = None,
    ):
        """
        Clean a correlation matrix in eigen-space.

        Parameters
        ----------
        correlation_matrix : torch.Tensor
            Correlation tensor with shape ``(batch, n_assets, n_assets)``.
        attributes : torch.Tensor, optional
            Optional auxiliary features concatenated to each eigenvalue
            channel. Accepted shapes are ``(batch, k)`` and
            ``(batch, n_assets, k)``.
        output_type : CorrelationTransformOutputType, optional
            Optional per-call override of requested output components.
        include_raw_eigenvectors : bool, default False
            Internal-use flag. When ``True``, the returned dictionary also
            contains a private ``'_raw_eigenvectors'`` entry with the
            orthogonal eigenvectors from the original decomposition.
        training : bool, optional
            Training flag passed to the recurrent transform.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            If one component is requested, returns that tensor directly. If
            multiple components are requested, returns a dictionary keyed by
            ``'correlation'``, ``'inverse_correlation'``, ``'eigenvalues'``,
            ``'eigenvectors'``, ``'inverse_eigenvalues'``.
        """
        if correlation_matrix.dim() != 3:
            raise ValueError("correlation_matrix must have shape (batch, n_assets, n_assets).")
        if correlation_matrix.shape[-2] != correlation_matrix.shape[-1]:
            raise ValueError("correlation_matrix must be square on the last two dimensions.")

        if attributes is not None:
            if attributes.dim() not in (2, 3):
                raise ValueError("attributes must have shape (batch, k) or (batch, n_assets, k).")
            if correlation_matrix.shape[0] != attributes.shape[0]:
                raise ValueError("Batch mismatch between correlation_matrix and attributes.")
            if attributes.dim() == 3 and correlation_matrix.shape[-1] != attributes.shape[1]:
                raise ValueError("Asset-dimension mismatch between correlation_matrix and attributes.")

        if not self.built:
            attr_shape = None if attributes is None else attributes.shape
            self.build(correlation_matrix.shape, attr_shape)
            self.to(device=correlation_matrix.device, dtype=correlation_matrix.dtype)
        is_training = resolve_training(self, training)

        components = self._resolve_output_components(output_type) if output_type is not None else list(self.output_components)
        need_correlation = "correlation" in components
        need_inverse_correlation = "inverse_correlation" in components
        need_eigenvalues = "eigenvalues" in components
        need_eigenvectors = "eigenvectors" in components
        need_inverse_eigenvalues = (
            "inverse_eigenvalues" in components
            or need_eigenvalues
            or need_correlation
            or need_inverse_correlation
            or need_eigenvectors
        )

        eigenvalues, eigenvectors = self.spectral_decomp(correlation_matrix)
        results = {}
        if include_raw_eigenvectors:
            results["_raw_eigenvectors"] = eigenvectors

        transformed_inverse_eigenvalues = None
        transformed_eigenvalues = None
        direct_eigenvectors = None

        if need_inverse_eigenvalues:
            eigenvalue_features = eigenvalues
            if attributes is not None:
                if attributes.dim() == 2:
                    n_assets = eigenvalues.shape[1]
                    attributes_tiled = attributes.unsqueeze(1).expand(attributes.shape[0], n_assets, attributes.shape[1])
                else:
                    attributes_tiled = attributes
                eigenvalue_features = torch.cat([eigenvalues, attributes_tiled], dim=-1)
            feature_width = eigenvalue_features.shape[-1]
            if self._feature_width is None:
                self._feature_width = int(feature_width)
            if int(feature_width) != self._feature_width:
                raise ValueError("Inconsistent eigenvalue feature width across calls.")
            transformed_inverse_eigenvalues = self.eigenvalue_transform(
                eigenvalue_features,
                training=is_training,
            )
            if "inverse_eigenvalues" in components:
                results["inverse_eigenvalues"] = transformed_inverse_eigenvalues.unsqueeze(-1)

        if need_eigenvalues or need_correlation or need_inverse_correlation or need_eigenvectors:
            inverse_eigs_work, inverse_dtype = ensure_float32(transformed_inverse_eigenvalues)
            eps = epsilon_for_dtype(inverse_eigs_work.dtype, self.epsilon).to(inverse_eigs_work.device)
            transformed_eigenvalues = restore_dtype(
                torch.reciprocal(torch.maximum(inverse_eigs_work, eps)),
                inverse_dtype,
            )
            if need_eigenvalues:
                results["eigenvalues"] = transformed_eigenvalues.unsqueeze(-1)

        if need_correlation or need_inverse_correlation or need_eigenvectors:
            direct_eigenvectors = self.eigenvector_rescaler((eigenvectors, transformed_eigenvalues))
        if need_eigenvectors:
            results["eigenvectors"] = direct_eigenvectors
        if need_correlation:
            results["correlation"] = self.correlation_product(transformed_eigenvalues, direct_eigenvectors)
        if need_inverse_correlation:
            results["inverse_correlation"] = self._exact_inverse_from_rescaled_eigensystem(
                direct_eigenvectors,
                transformed_inverse_eigenvalues,
            )
        if len(components) == 1 and not include_raw_eigenvectors:
            return results[components[0]]
        return results

    def get_config(self) -> dict:
        return {
            "recurrent_layer_sizes": self.recurrent_layer_sizes,
            "recurrent_cell": self.recurrent_cell,
            "recurrent_direction": self.recurrent_direction,
            "final_hidden_layer_sizes": self.final_hidden_layer_sizes,
            "final_hidden_activation": self.final_hidden_activation,
            "output_type": self._output_config,
            "epsilon": self.epsilon,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


class LagTransformLayer(nn.Module):
    """
    Module that applies a lag transformation to input financial time series.

    This module applies a non-linear transformation to financial returns that
    accounts for temporal dependencies and lag effects. The transformation uses
    learnable parameters to adaptively weight different time lags.

    Parameters
    ----------
    warm_start : bool, default True
        Whether to initialize trainable parameters near smooth deterministic
        profiles.
    name : str, optional
        Module name.
    eps : float, optional
        Base epsilon used in positivity constraints and safe divisions.
    variant : Literal["compact", "per_lag"], default "compact"
        Parameterization variant.
        - ``"compact"``: five-scalar parameterization with dynamic lookback
          support.
        - ``"per_lag"``: per-lag vectors with fixed lookback inferred at first
          build.

    Notes
    -----
    The transformation applied in both variants is::

        (alpha / (beta + eps)) * tanh(beta * R)
    """

    _ALLOWED_VARIANTS = {"compact", "per_lag"}

    def __init__(
        self,
        warm_start: bool = True,
        name: Optional[str] = None,
        eps: Optional[float] = None,
        variant: LagTransformVariant = "compact",
    ):
        """
        Initialize the lag-transform module.

        Parameters
        ----------
        warm_start : bool, default True
            If True, initialize parameters near smooth deterministic profiles.
            If False, use noisier random initializations.
        name : str, optional
            Module name.
        eps : float, optional
            Base epsilon used in positivity constraints and safe divisions.
        variant : Literal['compact', 'per_lag'], default 'compact'
            Lag parameterization mode.
        """
        super().__init__()
        variant = str(variant)
        if variant not in self._ALLOWED_VARIANTS:
            raise ValueError("variant must be one of {'compact', 'per_lag'}.")
        self.variant = variant
        self._eps_base = float(eps if eps is not None else 1e-7)
        self.warm_start = bool(warm_start)
        self._lookback_days: Optional[int] = None
        self._target = dict(c0=2.8, c1=0.20, c2=0.85, c3=0.50, c4=0.05)
        self.name = name
        self.built = False

    def _inv_softplus(self, y: float) -> float:
        y = float(y)
        if y <= 0.0:
            y = max(y, 1e-12)
        return math.log(math.expm1(y))

    def _add_param(self, target: float) -> nn.Parameter:
        mean_raw = self._inv_softplus(target - self._eps_base)
        if self.warm_start:
            value = torch.tensor(mean_raw, dtype=torch.float32)
        else:
            noise_scale = max(0.05 * abs(mean_raw), 1e-7)
            value = torch.randn((), dtype=torch.float32) * noise_scale + mean_raw
        return nn.Parameter(value)

    def _build_per_lag_profiles(self) -> Tuple[List[float], List[float]]:
        alpha: List[float] = []
        beta: List[float] = []
        for lag_idx in range(1, int(self._lookback_days) + 1):
            alpha.append(2.5 / (1.0 + 0.08 * (lag_idx - 1)))
            beta.append(0.25 + 0.70 * (1.0 - math.exp(-lag_idx / 8.0)))
        return alpha, beta

    def _init_vector_param(self, targets: List[float]) -> nn.Parameter:
        safe_targets = [max(t - self._eps_base, 1e-12) for t in targets]
        raw_targets = [self._inv_softplus(t) for t in safe_targets]
        if self.warm_start:
            value = torch.tensor(raw_targets, dtype=torch.float32)
        else:
            value = torch.randn((len(raw_targets),), dtype=torch.float32) * 0.1
        return nn.Parameter(value)

    def build(self, input_shape) -> None:
        """
        Build the lag-transform parameters.
        """
        lookback_from_shape = input_shape[-1]
        self._build_spec = (tuple(input_shape),)
        if self.variant == "compact":
            self._raw_c0 = self._add_param(self._target["c0"])
            self._raw_c1 = self._add_param(self._target["c1"])
            self._raw_c2 = self._add_param(self._target["c2"])
            self._raw_c3 = self._add_param(self._target["c3"])
            self._raw_c4 = self._add_param(self._target["c4"])
        else:
            if lookback_from_shape is None:
                raise ValueError("LagTransformLayer variant='per_lag' requires a static time dimension.")
            if self._lookback_days is None:
                self._lookback_days = int(lookback_from_shape)
            if int(lookback_from_shape) != self._lookback_days:
                raise ValueError(
                    "LagTransformLayer variant='per_lag' got incompatible time dimension at build: "
                    f"expected={self._lookback_days}, got={int(lookback_from_shape)}."
                )
            alpha_profile, beta_profile = self._build_per_lag_profiles()
            self._raw_alpha = self._init_vector_param(alpha_profile)
            self._raw_beta = self._init_vector_param(beta_profile)
        self.built = True

    def _pos(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + epsilon_for_dtype(x.dtype, self._eps_base).to(x.device)

    def _runtime_lookback_error(self, got: int) -> ValueError:
        return ValueError(
            "LagTransformLayer variant='per_lag' requires fixed lookback length: "
            f"expected={self._lookback_days}, got={got}."
        )

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        """
        Apply the lag transformation to returns.

        Parameters
        ----------
        R : torch.Tensor
            Returns tensor of shape ``(..., T)`` where the last axis is the
            lookback or time dimension.
            - In ``variant='compact'``, ``T`` can vary across calls.
            - In ``variant='per_lag'``, ``T`` must be fixed and equal to the
              first built time size.
            Sparse inputs are accepted and densified internally.

        Returns
        -------
        torch.Tensor
            Transformed returns with the same shape and dtype as ``R``.
        """
        R = ensure_dense_tensor(R)
        if not self.built:
            self.build(R.shape)
            param_dtype = torch.float32 if R.dtype in (torch.float16, torch.bfloat16) else R.dtype
            self.to(device=R.device, dtype=param_dtype)
        R_work, original_dtype = ensure_float32(R)
        dtype = R_work.dtype
        eps_tensor = epsilon_for_dtype(dtype, self._eps_base).to(R_work.device)

        if self.variant == "per_lag":
            if R_work.shape[-1] != self._lookback_days:
                raise self._runtime_lookback_error(int(R_work.shape[-1]))
            T = R_work.shape[-1]
            alpha = self._pos(self._raw_alpha).to(dtype=dtype, device=R_work.device)
            beta = self._pos(self._raw_beta).to(dtype=dtype, device=R_work.device)
            reshape = [1] * (R_work.dim() - 1) + [T]
            alpha_div_beta = (alpha / (beta + eps_tensor)).reshape(reshape)
            beta = beta.reshape(reshape)
            transformed = alpha_div_beta * torch.tanh(beta * R_work)
            return restore_dtype(transformed, original_dtype)

        T = R_work.shape[-1]
        t = torch.arange(1, T + 1, dtype=dtype, device=R_work.device).flip(0)
        c0 = self._pos(self._raw_c0).to(dtype=dtype, device=R_work.device)
        c1 = self._pos(self._raw_c1).to(dtype=dtype, device=R_work.device)
        c2 = self._pos(self._raw_c2).to(dtype=dtype, device=R_work.device)
        c3 = self._pos(self._raw_c3).to(dtype=dtype, device=R_work.device)
        c4 = self._pos(self._raw_c4).to(dtype=dtype, device=R_work.device)
        alpha = c0 * torch.pow(t, -c1)
        beta = c2 - c3 * torch.exp(-c4 * t)
        reshape = [1] * (R_work.dim() - 1) + [T]
        alpha_div_beta = (alpha / (beta + eps_tensor)).reshape(reshape)
        beta = beta.reshape(reshape)
        transformed = alpha_div_beta * torch.tanh(beta * R_work)
        return restore_dtype(transformed, original_dtype)

    def get_config(self) -> dict:
        return {
            "eps": self._eps_base,
            "warm_start": self.warm_start,
            "variant": self.variant,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


class RIEnetLayer(nn.Module):
    """
    Rotational Invariant Estimator (RIE) Network module for GMV portfolios.

    This module implements the compact network described in Bongiorno et al.
    for global minimum-variance (GMV) portfolio construction. The architecture
    couples rotational invariant estimators of the covariance matrix with
    recurrent neural networks in order to clean the eigen-spectrum and learn
    marginal volatilities in a parameter-efficient way.

    The module automatically scales daily returns by 252 by default and applies
    the following stages:

    - lag transformation with a five-parameter non-linearity,
    - sample covariance estimation and eigenvalue decomposition,
    - recurrent cleaning of eigenvalues,
    - dense transformation of marginal volatilities,
    - recombination into ``Sigma^{-1}`` followed by GMV weight normalization.

    Parameters
    ----------
    output_type : OutputType, default 'weights'
        Component(s) to return.
    recurrent_layer_sizes : Sequence[int], optional
        Hidden sizes of the recurrent cleaning block.
    std_hidden_layer_sizes : Sequence[int], optional
        Hidden sizes of the dense network acting on marginal volatilities.
    recurrent_cell : Literal['GRU', 'LSTM'], default 'GRU'
        Recurrent cell family used inside the eigenvalue cleaning block.
    recurrent_direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
        Direction used by the recurrent cleaning block.
    dimensional_features : Sequence[Literal['n_stocks', 'n_days', 'q', 'rsqrt_n_days']], optional
        Dimension-aware features concatenated to eigenvalues before recurrent
        cleaning.
    lag_transform_variant : Literal['compact', 'per_lag'], default 'compact'
        Lag transformation parameterization.
    normalize_transformed_variance : bool, default True
        Whether to normalize the transformed inverse volatilities so that the
        implied covariance diagonal is centred on 1.
    annualization_factor : float, default 252.0
        Factor used to scale daily returns.
    name : str, optional
        Module name.

    Notes
    -----
    Defaults replicate the compact RIE network optimised for GMV portfolios in
    the reference paper: a single bidirectional GRU layer with 16 units per
    direction and a dense marginal-volatility head with 8 hidden units.
    """

    def __init__(
        self,
        output_type: OutputType = "weights",
        recurrent_layer_sizes: Sequence[int] = (16,),
        std_hidden_layer_sizes: Sequence[int] = (8,),
        recurrent_cell: RecurrentCell = "GRU",
        recurrent_direction: RecurrentDirection = "bidirectional",
        dimensional_features: Sequence[DimensionalFeature] = ("n_stocks", "n_days", "q"),
        normalize_transformed_variance: bool = True,
        lag_transform_variant: LagTransformVariant = "compact",
        annualization_factor: float = 252.0,
        name: Optional[str] = None,
    ):
        """
        Initialize the RIEnet module.

        Parameters
        ----------
        output_type : OutputType, default 'weights'
            Requested output component(s). Allowed values are:
            ``'weights'``, ``'precision'``, ``'covariance'``,
            ``'correlation'``, ``'input_transformed'``, ``'eigenvalues'``,
            ``'eigenvectors'``, ``'transformed_std'`` and ``'all'``.
        recurrent_layer_sizes : Sequence[int], optional
            Hidden sizes of the recurrent cleaning block.
        std_hidden_layer_sizes : Sequence[int], optional
            Hidden sizes of the dense marginal-volatility block.
        recurrent_cell : Literal['GRU', 'LSTM'], default 'GRU'
            Recurrent cell type used in eigenvalue cleaning.
        recurrent_direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
            Direction mode of the recurrent cleaning block.
        dimensional_features : Sequence[Literal['n_stocks', 'n_days', 'q', 'rsqrt_n_days']], optional
            Additional features concatenated before eigenvalue cleaning.
        normalize_transformed_variance : bool, default True
            If True, rescales transformed inverse volatilities so that the
            implied covariance diagonal is centered around 1.
        lag_transform_variant : Literal['compact', 'per_lag'], default 'compact'
            Lag-transform parameterization.
        annualization_factor : float, default 252.0
            Factor used to scale daily returns for numerical stability.
        name : str, optional
            Module name.
        """
        super().__init__()
        allowed_outputs = (
            "weights",
            "precision",
            "covariance",
            "correlation",
            "input_transformed",
            "eigenvalues",
            "eigenvectors",
            "transformed_std",
        )
        self._output_config = output_type if isinstance(output_type, str) else list(output_type)
        if isinstance(output_type, str):
            components = list(allowed_outputs) if output_type == "all" else [output_type]
        else:
            if not output_type:
                raise ValueError("output_type cannot be an empty sequence.")
            components = []
            for entry in output_type:
                if entry == "all":
                    components.extend(allowed_outputs)
                else:
                    components.append(entry)
        deduped: List[str] = []
        for entry in components:
            if entry not in allowed_outputs:
                raise ValueError("Invalid output_type entry.")
            if entry not in deduped:
                deduped.append(entry)
        self.output_components = tuple(deduped)
        self.output_type = deduped[0] if len(deduped) == 1 else tuple(deduped)

        self._need_precision = "precision" in self.output_components
        self._need_covariance = "covariance" in self.output_components
        self._need_correlation = "correlation" in self.output_components
        self._need_weights = "weights" in self.output_components
        self._need_eigenvalues = "eigenvalues" in self.output_components
        self._need_eigenvectors = "eigenvectors" in self.output_components
        self._need_transformed_std = "transformed_std" in self.output_components
        self._need_structural_outputs = self._need_precision or self._need_covariance or self._need_correlation or self._need_weights
        self._need_spectral_outputs = self._need_eigenvalues or self._need_eigenvectors or self._need_transformed_std
        self._need_pipeline_outputs = self._need_structural_outputs or self._need_spectral_outputs
        self._need_inverse_std_branch = self._need_precision or self._need_covariance or self._need_weights or self._need_transformed_std
        self._need_spectral_branch = (
            self._need_precision
            or self._need_covariance
            or self._need_correlation
            or self._need_weights
            or self._need_eigenvalues
            or self._need_eigenvectors
        )

        if recurrent_layer_sizes is None:
            raise ValueError("recurrent_layer_sizes cannot be None; pass a non-empty sequence of positive integers.")
        recurrent_layer_sizes = list(recurrent_layer_sizes)
        if not recurrent_layer_sizes:
            raise ValueError("recurrent_layer_sizes must contain at least one positive integer")
        if any(size <= 0 for size in recurrent_layer_sizes):
            raise ValueError("recurrent_layer_sizes must contain positive integers")

        if std_hidden_layer_sizes is None:
            raise ValueError("std_hidden_layer_sizes cannot be None; pass a non-empty sequence of positive integers.")
        std_hidden_layer_sizes = list(std_hidden_layer_sizes)
        if not std_hidden_layer_sizes:
            raise ValueError("std_hidden_layer_sizes must contain at least one positive integer")
        if any(size <= 0 for size in std_hidden_layer_sizes):
            raise ValueError("std_hidden_layer_sizes must contain positive integers")

        normalized_cell = recurrent_cell.strip().upper()
        if normalized_cell not in {"GRU", "LSTM"}:
            raise ValueError("recurrent_cell must be either 'GRU' or 'LSTM'")
        normalized_direction = recurrent_direction.strip().lower()
        if normalized_direction not in {"bidirectional", "forward", "backward"}:
            raise ValueError("recurrent_direction must be 'bidirectional', 'forward', or 'backward'.")

        if dimensional_features is None:
            dimensional_features = ("n_stocks", "n_days", "q")
        dimensional_features = list(dimensional_features)
        allowed_features = {"n_stocks", "n_days", "q", "rsqrt_n_days"}
        invalid_features = [feature for feature in dimensional_features if feature not in allowed_features]
        if invalid_features:
            raise ValueError(
                "dimensional_features entries must be in "
                "{'n_stocks', 'n_days', 'q', 'rsqrt_n_days'}; "
                f"got invalid entries: {invalid_features}."
            )

        self._std_hidden_layer_sizes = std_hidden_layer_sizes
        self._recurrent_layer_sizes = recurrent_layer_sizes
        self._recurrent_model = normalized_cell
        self._direction = normalized_direction
        self._dimensional_features = list(dict.fromkeys(dimensional_features))
        self._annualization_factor = float(annualization_factor)
        self._normalize_variance = bool(normalize_transformed_variance)
        self._lag_transform_variant = lag_transform_variant
        self.name = name
        self.built = False

        self._build_layers()

    def _build_layers(self) -> None:
        self.lag_transform = LagTransformLayer(
            variant=self._lag_transform_variant,
            warm_start=True,
            name=f"{self.name}_lag_transform",
        )
        self.std_layer = StandardDeviationLayer(axis=-1, name=f"{self.name}_std")
        self.covariance_layer = CovarianceLayer(expand_dims=False, normalize=True, name=f"{self.name}_covariance")
        self.dimension_aware = DimensionAwareLayer(features=self._dimensional_features, name=f"{self.name}_dimension_aware")
        self.correlation_eigen_transform = None
        if self._need_spectral_branch:
            self.correlation_eigen_transform = CorrelationEigenTransformLayer(
                recurrent_layer_sizes=tuple(self._recurrent_layer_sizes),
                recurrent_cell=self._recurrent_model,
                recurrent_direction=self._direction,
                final_hidden_layer_sizes=(),
                output_type="correlation",
                name=f"{self.name}_corr_eigen_transform",
            )
        self.std_transform = None
        if self._need_inverse_std_branch:
            self.std_transform = DeepLayer(
                hidden_layer_sizes=self._std_hidden_layer_sizes + [1],
                last_activation="softplus",
                name=f"{self.name}_std_transform",
            )
        self.std_normalization = None
        if self._normalize_variance and self._need_inverse_std_branch:
            self.std_normalization = CustomNormalizationLayer(
                axis=-2,
                mode="inverse",
                inverse_power=2.0,
                name=f"{self.name}_std_norm",
            )
        self.outer_product = None
        if self._need_precision or self._need_covariance or self._need_weights:
            self.outer_product = CovarianceLayer(
                normalize=False,
                name=f"{self.name}_inverse_scale_outer",
            )

    def build(self, input_shape) -> None:
        """
        Build submodules once input dimensionality is known.
        """
        self._build_spec = (tuple(input_shape),)
        if self._need_spectral_branch and self.correlation_eigen_transform is not None:
            attributes_shape = (input_shape[0], input_shape[1], len(self._dimensional_features))
            covariance_shape = (input_shape[0], input_shape[1], input_shape[1])
            self.correlation_eigen_transform.build(covariance_shape, attributes_shape)
        self.built = True

    def _normalize_raw_weights(
        self,
        raw_weights: torch.Tensor,
        original_dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        epsilon = epsilon_for_dtype(raw_weights.dtype, 1e-7).to(raw_weights.device)
        denom = raw_weights.sum(dim=-1, keepdim=True)
        sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
        safe_denom = torch.where(denom.abs() < epsilon, sign * epsilon, denom)
        return restore_dtype((raw_weights / safe_denom).unsqueeze(-1), original_dtype)

    def _exact_weights_from_rescaled_eigensystem(
        self,
        direct_eigenvectors: torch.Tensor,
        inverse_eigenvalues: torch.Tensor,
        inverse_std: torch.Tensor,
    ) -> torch.Tensor:
        vectors_work, original_dtype = ensure_float32(torch.as_tensor(direct_eigenvectors))
        dtype = vectors_work.dtype
        inverse_eigenvalues = torch.as_tensor(inverse_eigenvalues, dtype=dtype, device=vectors_work.device).reshape(vectors_work.shape[:-1])
        inverse_std = torch.as_tensor(inverse_std, dtype=dtype, device=vectors_work.device).reshape(vectors_work.shape[:-1])
        row_norm_sq = vectors_work.square().sum(dim=-1)
        eps = epsilon_for_dtype(dtype, 1e-7).to(vectors_work.device)
        inverse_row_norm_sq = torch.reciprocal(torch.maximum(row_norm_sq, eps))
        spectral_rhs = torch.matmul(vectors_work.transpose(-1, -2), (inverse_row_norm_sq * inverse_std).unsqueeze(-1)).squeeze(-1)
        spectral_rhs = inverse_eigenvalues * spectral_rhs
        raw_weights = inverse_std * inverse_row_norm_sq * torch.matmul(vectors_work, spectral_rhs.unsqueeze(-1)).squeeze(-1)
        return self._normalize_raw_weights(raw_weights, original_dtype)

    def _exact_weights_from_raw_eigensystem(
        self,
        orthogonal_eigenvectors: torch.Tensor,
        inverse_eigenvalues: torch.Tensor,
        inverse_std: torch.Tensor,
    ) -> torch.Tensor:
        vectors_work, original_dtype = ensure_float32(torch.as_tensor(orthogonal_eigenvectors))
        dtype = vectors_work.dtype
        inverse_eigenvalues = torch.as_tensor(inverse_eigenvalues, dtype=dtype, device=vectors_work.device).reshape(vectors_work.shape[:-1])
        inverse_std = torch.as_tensor(inverse_std, dtype=dtype, device=vectors_work.device).reshape(vectors_work.shape[:-1])
        eps = epsilon_for_dtype(dtype, 1e-7).to(vectors_work.device)
        eigenvalues = torch.reciprocal(torch.maximum(inverse_eigenvalues, eps))
        diagonal_scale_sq = torch.einsum("...ij,...j,...ij->...i", vectors_work, eigenvalues, vectors_work)
        diagonal_scale = torch.sqrt(torch.maximum(diagonal_scale_sq, eps))
        scaled_inverse_std = diagonal_scale * inverse_std
        spectral_rhs = torch.matmul(vectors_work.transpose(-1, -2), scaled_inverse_std.unsqueeze(-1)).squeeze(-1)
        raw_weights = scaled_inverse_std * torch.matmul(vectors_work, (inverse_eigenvalues * spectral_rhs).unsqueeze(-1)).squeeze(-1)
        return self._normalize_raw_weights(raw_weights, original_dtype)

    def _weights_from_inverse_correlation(self, inverse_correlation: torch.Tensor, inverse_std: torch.Tensor) -> torch.Tensor:
        inverse_corr_work, original_dtype = ensure_float32(inverse_correlation)
        inverse_std = torch.as_tensor(inverse_std, dtype=inverse_corr_work.dtype, device=inverse_corr_work.device).reshape(inverse_corr_work.shape[:-1])
        raw_weights = inverse_std * torch.matmul(inverse_corr_work, inverse_std.unsqueeze(-1)).squeeze(-1)
        return self._normalize_raw_weights(raw_weights, original_dtype)

    def _precision_outputs(
        self,
        inverse_correlation: torch.Tensor,
        inverse_std: torch.Tensor,
        include_weights: bool,
    ) -> dict[str, torch.Tensor]:
        inverse_volatility_matrix = self.outer_product(inverse_std)
        outputs = {"precision": inverse_correlation * inverse_volatility_matrix}
        if include_weights:
            outputs["weights"] = self._weights_from_inverse_correlation(inverse_correlation, inverse_std)
        return outputs

    def forward(self, inputs: torch.Tensor, training: Optional[bool] = None):
        """
        Execute the full RIEnet pipeline.

        Parameters
        ----------
        inputs : torch.Tensor
            Daily returns tensor with shape ``(batch_size, n_stocks, n_days)``.
        training : bool, optional
            Training flag forwarded to stochastic submodules.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            If one output is requested, returns a single tensor. If multiple
            outputs are requested, returns a dictionary. Components and shapes:
            - ``weights``: ``(batch, n_stocks, 1)``
            - ``precision``: ``(batch, n_stocks, n_stocks)``
            - ``covariance``: ``(batch, n_stocks, n_stocks)``
            - ``correlation``: ``(batch, n_stocks, n_stocks)``
            - ``eigenvalues``: ``(batch, n_stocks, 1)``
            - ``eigenvectors``: ``(batch, n_stocks, n_stocks)``
            - ``transformed_std``: ``(batch, n_stocks, 1)``
            - ``input_transformed``: ``(batch, n_stocks, n_days)``
        """
        inputs = ensure_dense_tensor(inputs)
        if inputs.dim() != 3:
            raise ValueError("inputs must have shape (batch_size, n_stocks, n_days).")
        if not torch.is_floating_point(inputs):
            raise TypeError("inputs must be a floating-point tensor.")
        if 0 in inputs.shape:
            raise ValueError("inputs must have non-empty batch, asset, and time dimensions.")
        if not torch.isfinite(inputs).all():
            raise ValueError("inputs must contain only finite values.")

        if not self.built:
            self.build(inputs.shape)
            self.to(device=inputs.device, dtype=inputs.dtype)
        is_training = resolve_training(self, training)

        need_precision = self._need_precision
        need_covariance = self._need_covariance
        need_correlation = self._need_correlation
        need_weights = self._need_weights
        need_eigenvalues = self._need_eigenvalues
        need_eigenvectors = self._need_eigenvectors
        need_transformed_std = self._need_transformed_std
        need_pipeline_outputs = self._need_pipeline_outputs

        scaled_inputs = inputs * self._annualization_factor
        input_transformed = self.lag_transform(scaled_inputs)
        results = {}
        if "input_transformed" in self.output_components:
            results["input_transformed"] = input_transformed
        if not need_pipeline_outputs:
            return results[self.output_components[0]] if len(self.output_components) == 1 else results

        std, mean = self.std_layer(input_transformed)
        need_inverse_std = need_precision or need_covariance or need_weights or need_transformed_std
        need_reciprocal_std = need_covariance or need_transformed_std
        transformed_inverse_std = None
        std_for_structural = None
        transformed_std = None
        if need_inverse_std:
            transformed_inverse_std = self.std_transform(std)
            std_for_structural = transformed_inverse_std
            if self.std_normalization is not None:
                std_for_structural = self.std_normalization(transformed_inverse_std)
            if need_reciprocal_std:
                std_work, std_original_dtype = ensure_float32(std_for_structural)
                std_eps = epsilon_for_dtype(std_work.dtype, 1e-7).to(std_work.device)
                transformed_std = restore_dtype(torch.reciprocal(torch.maximum(std_work, std_eps)), std_original_dtype)
        if need_transformed_std:
            results["transformed_std"] = transformed_std
        if not self._need_spectral_branch:
            return results[self.output_components[0]] if len(self.output_components) == 1 else results

        zscores = (input_transformed - mean) / std
        correlation_matrix = self.covariance_layer(zscores)
        attributes = None
        if self._dimensional_features:
            attributes = self.dimension_aware([zscores, correlation_matrix])

        spectral_components: List[str] = []
        need_fast_weight_path = need_weights and not (need_precision or need_covariance or need_correlation or need_eigenvectors)
        need_legacy_weight_eigensystem = need_weights and not need_precision and not need_fast_weight_path
        if need_eigenvectors or need_legacy_weight_eigensystem:
            spectral_components.append("eigenvectors")
        if need_eigenvalues:
            spectral_components.append("eigenvalues")
        if need_fast_weight_path or need_legacy_weight_eigensystem:
            spectral_components.append("inverse_eigenvalues")
        if need_covariance or need_correlation:
            spectral_components.append("correlation")
        if need_precision:
            spectral_components.append("inverse_correlation")
        spectral_components = list(dict.fromkeys(spectral_components))

        spectral_outputs = self.correlation_eigen_transform(
            correlation_matrix,
            attributes=attributes,
            output_type=spectral_components,
            include_raw_eigenvectors=need_fast_weight_path,
            training=is_training,
        )
        spectral_results = spectral_outputs if isinstance(spectral_outputs, dict) else {spectral_components[0]: spectral_outputs}
        eigenvectors = spectral_results.get("eigenvectors")
        raw_eigenvectors = spectral_results.get("_raw_eigenvectors")
        transformed_inverse_eigenvalues = spectral_results.get("inverse_eigenvalues")
        if transformed_inverse_eigenvalues is not None:
            transformed_inverse_eigenvalues = transformed_inverse_eigenvalues.squeeze(-1)
        cleaned_correlation = spectral_results.get("correlation")

        if need_eigenvectors:
            results["eigenvectors"] = eigenvectors
        if need_eigenvalues:
            results["eigenvalues"] = spectral_results["eigenvalues"]
        inverse_correlation = spectral_results.get("inverse_correlation")

        if need_covariance:
            volatility_matrix = self.outer_product(transformed_std)
            results["covariance"] = cleaned_correlation * volatility_matrix
        if need_correlation:
            results["correlation"] = cleaned_correlation
        if need_precision:
            results.update(self._precision_outputs(inverse_correlation, std_for_structural, need_weights))
        if need_weights and not need_precision:
            if need_fast_weight_path:
                results["weights"] = self._exact_weights_from_raw_eigensystem(
                    raw_eigenvectors,
                    transformed_inverse_eigenvalues,
                    std_for_structural,
                )
            else:
                results["weights"] = self._exact_weights_from_rescaled_eigensystem(
                    eigenvectors,
                    transformed_inverse_eigenvalues,
                    std_for_structural,
                )
        return results[self.output_components[0]] if len(self.output_components) == 1 else results

    def get_config(self) -> dict:
        return {
            "output_type": self._output_config,
            "recurrent_layer_sizes": list(self._recurrent_layer_sizes),
            "std_hidden_layer_sizes": list(self._std_hidden_layer_sizes),
            "recurrent_cell": self._recurrent_model,
            "recurrent_direction": self._direction,
            "dimensional_features": list(self._dimensional_features),
            "normalize_transformed_variance": self._normalize_variance,
            "lag_transform_variant": self._lag_transform_variant,
            "annualization_factor": self._annualization_factor,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

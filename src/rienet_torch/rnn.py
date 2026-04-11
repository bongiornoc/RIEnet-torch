"""
Keras-style dense and recurrent primitives used internally by RIEnet.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def get_activation(name: Optional[str]) -> Callable[[torch.Tensor], torch.Tensor]:
    if name in (None, "linear"):
        return lambda x: x
    if name == "softplus":
        return F.softplus
    if name == "sigmoid":
        return torch.sigmoid
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=0.2)
    raise ValueError(f"Unsupported activation: {name}")


def keras_init_kernel(shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return torch.empty(shape, device=device, dtype=dtype).uniform_(-limit, limit)


def keras_init_orthogonal(shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows, cols = shape
    matrix = torch.empty((rows, cols), device=device, dtype=dtype)
    nn.init.orthogonal_(matrix)
    return matrix


def dropout_mask_like(x: torch.Tensor, rate: float) -> torch.Tensor:
    keep_prob = 1.0 - rate
    if keep_prob <= 0.0:
        return torch.zeros_like(x)
    mask = torch.empty_like(x).bernoulli_(keep_prob)
    return mask / keep_prob


def resolve_training(module: nn.Module, training: Optional[bool]) -> bool:
    return module.training if training is None else bool(training)


class KerasDense(nn.Module):
    def __init__(
        self,
        units: int,
        *,
        activation: str = "linear",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        name: Optional[str] = None,
    ):
        super().__init__()
        self.units = int(units)
        self.activation_name = activation
        self.activation = get_activation(activation)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = kernel_initializer
        self.name = name
        self.input_dim: Optional[int] = None
        self.kernel: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        self.built = False

    def build(self, input_shape) -> None:
        input_dim = int(input_shape[-1])
        self.input_dim = input_dim
        device = torch.device("cpu")
        dtype = torch.float32
        kernel = keras_init_kernel((input_dim, self.units), device, dtype)
        self.kernel = nn.Parameter(kernel)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((self.units,), device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.built = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.built:
            self.build(inputs.shape)
            self.to(device=inputs.device, dtype=inputs.dtype)
        outputs = torch.matmul(inputs, self.kernel)
        if self.bias is not None:
            outputs = outputs + self.bias
        return self.activation(outputs)


class KerasGRULayer(nn.Module):
    def __init__(
        self,
        units: int,
        *,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        return_sequences: bool = True,
        go_backwards: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.units = int(units)
        self.dropout = float(dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.return_sequences = bool(return_sequences)
        self.go_backwards = bool(go_backwards)
        self.name = name
        self.input_dim: Optional[int] = None
        self.kernel: Optional[nn.Parameter] = None
        self.recurrent_kernel: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        self.built = False

    def build(self, input_shape) -> None:
        input_dim = int(input_shape[-1])
        self.input_dim = input_dim
        device = torch.device("cpu")
        dtype = torch.float32
        self.kernel = nn.Parameter(keras_init_kernel((input_dim, 3 * self.units), device, dtype))
        self.recurrent_kernel = nn.Parameter(keras_init_orthogonal((self.units, 3 * self.units), device, dtype))
        self.bias = nn.Parameter(torch.zeros((2, 3 * self.units), device=device, dtype=dtype))
        self.built = True

    def forward(self, inputs: torch.Tensor, training: Optional[bool] = None) -> torch.Tensor:
        if not self.built:
            self.build(inputs.shape)
            self.to(device=inputs.device, dtype=inputs.dtype)

        batch, timesteps, _ = inputs.shape
        h = torch.zeros((batch, self.units), dtype=inputs.dtype, device=inputs.device)
        is_training = resolve_training(self, training)

        input_masks = None
        recurrent_masks = None
        if is_training and self.dropout > 0.0:
            input_masks = tuple(dropout_mask_like(inputs[:, 0, :], self.dropout) for _ in range(3))
        if is_training and self.recurrent_dropout > 0.0:
            recurrent_masks = tuple(dropout_mask_like(h, self.recurrent_dropout) for _ in range(3))

        time_range = range(timesteps - 1, -1, -1) if self.go_backwards else range(timesteps)
        input_bias, recurrent_bias = self.bias[0], self.bias[1]
        k_z, k_r, k_h = torch.split(self.kernel, self.units, dim=1)
        rk_z, rk_r, rk_h = torch.split(self.recurrent_kernel, self.units, dim=1)
        b_iz, b_ir, b_ih = torch.split(input_bias, self.units, dim=0)
        b_rz, b_rr, b_rh = torch.split(recurrent_bias, self.units, dim=0)

        if input_masks is None:
            input_all = torch.matmul(inputs, self.kernel) + input_bias.view(1, 1, -1)
            x_z_seq, x_r_seq, x_h_seq = torch.split(input_all, self.units, dim=-1)
        else:
            x_z_seq = torch.matmul(inputs * input_masks[0].unsqueeze(1), k_z) + b_iz
            x_r_seq = torch.matmul(inputs * input_masks[1].unsqueeze(1), k_r) + b_ir
            x_h_seq = torch.matmul(inputs * input_masks[2].unsqueeze(1), k_h) + b_ih

        outputs = inputs.new_empty((batch, timesteps, self.units))

        for step_index, t in enumerate(time_range):
            x_z = x_z_seq[:, t, :]
            x_r = x_r_seq[:, t, :]
            x_h = x_h_seq[:, t, :]

            if recurrent_masks is None:
                recurrent_all = torch.matmul(h, self.recurrent_kernel) + recurrent_bias
                recurrent_z, recurrent_r, recurrent_h = torch.split(recurrent_all, self.units, dim=-1)
            else:
                recurrent_z = torch.matmul(h * recurrent_masks[0], rk_z) + b_rz
                recurrent_r = torch.matmul(h * recurrent_masks[1], rk_r) + b_rr
                recurrent_h = torch.matmul(h * recurrent_masks[2], rk_h) + b_rh

            z = torch.sigmoid(x_z + recurrent_z)
            r = torch.sigmoid(x_r + recurrent_r)
            hh = torch.tanh(x_h + r * recurrent_h)
            h = z * h + (1.0 - z) * hh
            outputs[:, step_index, :] = h
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


class KerasLSTMLayer(nn.Module):
    def __init__(
        self,
        units: int,
        *,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        return_sequences: bool = True,
        go_backwards: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.units = int(units)
        self.dropout = float(dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.return_sequences = bool(return_sequences)
        self.go_backwards = bool(go_backwards)
        self.name = name
        self.input_dim: Optional[int] = None
        self.kernel: Optional[nn.Parameter] = None
        self.recurrent_kernel: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        self.built = False

    def build(self, input_shape) -> None:
        input_dim = int(input_shape[-1])
        self.input_dim = input_dim
        device = torch.device("cpu")
        dtype = torch.float32
        self.kernel = nn.Parameter(keras_init_kernel((input_dim, 4 * self.units), device, dtype))
        self.recurrent_kernel = nn.Parameter(keras_init_orthogonal((self.units, 4 * self.units), device, dtype))
        bias = torch.zeros((4 * self.units,), device=device, dtype=dtype)
        bias[self.units : 2 * self.units] = 1.0
        self.bias = nn.Parameter(bias)
        self.built = True

    def forward(self, inputs: torch.Tensor, training: Optional[bool] = None) -> torch.Tensor:
        if not self.built:
            self.build(inputs.shape)
            self.to(device=inputs.device, dtype=inputs.dtype)

        batch, timesteps, _ = inputs.shape
        h = torch.zeros((batch, self.units), dtype=inputs.dtype, device=inputs.device)
        c = torch.zeros((batch, self.units), dtype=inputs.dtype, device=inputs.device)
        is_training = resolve_training(self, training)

        input_masks = None
        recurrent_masks = None
        if is_training and self.dropout > 0.0:
            input_masks = tuple(dropout_mask_like(inputs[:, 0, :], self.dropout) for _ in range(4))
        if is_training and self.recurrent_dropout > 0.0:
            recurrent_masks = tuple(dropout_mask_like(h, self.recurrent_dropout) for _ in range(4))

        outputs = []
        time_range = range(timesteps - 1, -1, -1) if self.go_backwards else range(timesteps)
        k_i, k_f, k_c, k_o = torch.split(self.kernel, self.units, dim=1)
        r_i, r_f, r_c, r_o = torch.split(self.recurrent_kernel, self.units, dim=1)
        b_i, b_f, b_c, b_o = torch.split(self.bias, self.units, dim=0)

        for t in time_range:
            x_t = inputs[:, t, :]
            x_i = torch.matmul(x_t if input_masks is None else x_t * input_masks[0], k_i) + b_i
            x_f = torch.matmul(x_t if input_masks is None else x_t * input_masks[1], k_f) + b_f
            x_c = torch.matmul(x_t if input_masks is None else x_t * input_masks[2], k_c) + b_c
            x_o = torch.matmul(x_t if input_masks is None else x_t * input_masks[3], k_o) + b_o

            h_i = h if recurrent_masks is None else h * recurrent_masks[0]
            h_f = h if recurrent_masks is None else h * recurrent_masks[1]
            h_c = h if recurrent_masks is None else h * recurrent_masks[2]
            h_o = h if recurrent_masks is None else h * recurrent_masks[3]

            i = torch.sigmoid(x_i + torch.matmul(h_i, r_i))
            f = torch.sigmoid(x_f + torch.matmul(h_f, r_f))
            c = f * c + i * torch.tanh(x_c + torch.matmul(h_c, r_c))
            o = torch.sigmoid(x_o + torch.matmul(h_o, r_o))
            h = o * torch.tanh(c)
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        return outputs[:, -1, :]


class KerasBidirectional(nn.Module):
    def __init__(self, layer: nn.Module, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.forward_layer = layer
        backward = type(layer)(
            units=layer.units,
            dropout=layer.dropout,
            recurrent_dropout=layer.recurrent_dropout,
            return_sequences=layer.return_sequences,
            go_backwards=True,
            name=f"backward_{layer.name}",
        )
        self.backward_layer = backward
        self.merge_mode = "concat"

    def build(self, input_shape) -> None:
        if hasattr(self.forward_layer, "build"):
            self.forward_layer.build(input_shape)
        if hasattr(self.backward_layer, "build"):
            self.backward_layer.build(input_shape)

    def forward(self, sequences: torch.Tensor, training: Optional[bool] = None) -> torch.Tensor:
        is_training = resolve_training(self, training)
        y = self.forward_layer(sequences, training=is_training)
        y_rev = self.backward_layer(sequences, training=is_training)
        y_rev = torch.flip(y_rev, dims=[1])
        return torch.cat([y, y_rev], dim=-1)

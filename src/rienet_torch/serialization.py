"""
Serialization helpers for RIEnet Torch.

This module provides PyTorch-side helpers based on ``state_dict`` plus
optional ``get_config``/``from_config`` roundtrips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import torch


def _dummy_tensor_from_shape(shape: Any) -> torch.Tensor | None:
    if shape is None:
        return None
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    if isinstance(shape, (list, tuple)):
        return torch.zeros(tuple(shape), dtype=torch.float32)
    raise TypeError(f"Unsupported build shape spec: {shape!r}")


def _materialize_from_build_spec(module: torch.nn.Module, build_spec: Any) -> None:
    if build_spec is None:
        return
    dummy_args = [_dummy_tensor_from_shape(shape) for shape in build_spec]
    while dummy_args and dummy_args[-1] is None:
        dummy_args.pop()
    with torch.no_grad():
        module(*dummy_args)


def save_module(module: torch.nn.Module, path: str | Path) -> None:
    """
    Save a RIEnet module together with config and lazy-build metadata.

    Parameters
    ----------
    module : torch.nn.Module
        Module to serialize. If it exposes ``get_config`` the config is stored
        alongside the ``state_dict``.
    path : str or pathlib.Path
        Destination path passed to ``torch.save``.
    """
    payload: dict[str, Any] = {
        "state_dict": module.state_dict(),
    }
    if hasattr(module, "get_config"):
        payload["config"] = module.get_config()
        payload["class_name"] = module.__class__.__name__
    build_spec = getattr(module, "_build_spec", None)
    if build_spec is not None:
        payload["build_spec"] = build_spec
    torch.save(payload, path)


def load_module(
    cls: Type[torch.nn.Module],
    path: str | Path,
    *,
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load a RIEnet module previously saved with :func:`save_module`.

    Parameters
    ----------
    cls : type[torch.nn.Module]
        Module class to instantiate.
    path : str or pathlib.Path
        File created by :func:`save_module`.
    strict : bool, default True
        Passed to ``load_state_dict``.

    Returns
    -------
    torch.nn.Module
        A reconstructed module with config, build spec, and parameters restored.
    """
    payload = torch.load(path, map_location="cpu")
    if "config" in payload and hasattr(cls, "from_config"):
        module = cls.from_config(payload["config"])
    else:
        module = cls()
    if "build_spec" in payload:
        _materialize_from_build_spec(module, payload["build_spec"])
    module.load_state_dict(payload["state_dict"], strict=strict)
    return module

"""
Public lag-transformation module for RIEnet Torch.

This module re-exports ``LagTransformLayer`` and ``LagTransformVariant`` from
``rienet_torch.trainable_layers`` so that the standalone lag transform can be
imported from either location.
"""

from .trainable_layers import LagTransformLayer, LagTransformVariant

__all__ = [
    "LagTransformLayer",
    "LagTransformVariant",
]

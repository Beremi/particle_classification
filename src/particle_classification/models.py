"""Compatibility façade for models that now live with their experiments."""

from .experiments.baseline.model import XYInvariantParticleNet, XYInvariantParticleNetConfig
from .experiments.neural_separator.model import (
    EdgeTrackNetTiny,
    EdgeTrackNetTinyConfig,
    bridge_edge_passes,
    connected_components_from_edges,
    edge_aux_features,
    prepare_edge_attr,
    relabel_components,
    split_suspicious_components,
)

__all__ = [
    "XYInvariantParticleNet",
    "XYInvariantParticleNetConfig",
    "EdgeTrackNetTiny",
    "EdgeTrackNetTinyConfig",
    "bridge_edge_passes",
    "connected_components_from_edges",
    "edge_aux_features",
    "prepare_edge_attr",
    "relabel_components",
    "split_suspicious_components",
]

"""Compatibility façade for voxel autoencoder experiments."""

from .experiments.autoencoders.voxel import *  # noqa: F401,F403
from .experiments.autoencoders.voxel import (
    empty_voxel_metadata,
    ensure_voxel_channel,
    evaluate_voxel_ae,
    flush_voxel_chunk,
    init_voxel_weights,
    load_voxel_manifest,
)
from .experiments.autoencoders.voxel import __all__ as _CANONICAL_ALL

__all__ = [
    *_CANONICAL_ALL,
    "empty_voxel_metadata",
    "ensure_voxel_channel",
    "evaluate_voxel_ae",
    "flush_voxel_chunk",
    "init_voxel_weights",
    "load_voxel_manifest",
]

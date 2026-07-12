"""Compatibility façade for canonical voxel autoencoder experiments."""

from .experiments.autoencoders.canonical_voxel import *  # noqa: F401,F403
from .experiments.autoencoders.canonical_voxel import (
    VoxelAEStage,
    _load_canonical_checkpoint,
    _render_rows,
    dilated_support,
    evaluate_canonical_model,
    evaluate_energy_l2_model,
    hardmine_reconstruction_loss,
    hit_count_bucket,
    normalized_coordinate_grid,
    random_augmentation_transform,
    relative_tensor_l2_loss,
    render_canonical_latent_pairs_main,
    render_canonical_voxel_gallery_main,
    train_canonical_voxel_ae_main,
    transform_raw_to_params,
    write_hard_mine_selection,
    write_hardmine_l2_training_report,
)
from .experiments.autoencoders.canonical_voxel import __all__ as _CANONICAL_ALL

__all__ = [
    *_CANONICAL_ALL,
    "VoxelAEStage",
    "_load_canonical_checkpoint",
    "_render_rows",
    "dilated_support",
    "evaluate_canonical_model",
    "evaluate_energy_l2_model",
    "hardmine_reconstruction_loss",
    "hit_count_bucket",
    "normalized_coordinate_grid",
    "random_augmentation_transform",
    "relative_tensor_l2_loss",
    "render_canonical_latent_pairs_main",
    "render_canonical_voxel_gallery_main",
    "train_canonical_voxel_ae_main",
    "transform_raw_to_params",
    "write_hard_mine_selection",
    "write_hardmine_l2_training_report",
]

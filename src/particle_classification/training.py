"""Compatibility façade for the relocated XY-invariant baseline helpers."""

from .experiments.baseline.training import (
    CandidateDataset,
    candidate_features,
    circular_pose_loss,
    collate_candidates,
    load_bootstrap_candidates,
    rotate_candidates,
    train_baseline_from_config,
    xy_invariance_loss,
)

__all__ = [
    "CandidateDataset",
    "candidate_features",
    "circular_pose_loss",
    "collate_candidates",
    "load_bootstrap_candidates",
    "rotate_candidates",
    "train_baseline_from_config",
    "xy_invariance_loss",
]

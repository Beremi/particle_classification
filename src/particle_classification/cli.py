"""Compatibility façade for historical ``particle_classification.cli`` imports.

New console entry points import the grouped modules under
``particle_classification.commands`` directly.
"""

from .commands.data import (
    extract_raw_main,
    data_index_main,
    build_candidates_main,
)
from .commands.dbscan import (
    tune_dbscan_main,
    build_particles_main,
    benchmark_clustering_main,
    audit_particle_shards_main,
    audit_particle_continuity_main,
)
from .commands.neural_separator import (
    build_edge_training_set_main,
    build_edge_mixed_set_main,
    build_edge_curriculum_set_main,
    train_edge_tracknet_main,
    run_edge_experiment_main,
    run_edge_replacement_search_main,
    evaluate_phase1_main,
)
from .commands.autoencoders import (
    build_phase2_dataset_main,
    train_phase2_sweep_main,
    evaluate_phase2_main,
    generate_phase2_report_main,
    train_canonical_voxel_ae_main,
    render_canonical_voxel_gallery_main,
    render_canonical_latent_pairs_main,
)
from .commands.baseline import (
    train_baseline_main,
)

__all__ = [
    "extract_raw_main",
    "data_index_main",
    "build_candidates_main",
    "tune_dbscan_main",
    "build_particles_main",
    "benchmark_clustering_main",
    "audit_particle_shards_main",
    "audit_particle_continuity_main",
    "build_edge_training_set_main",
    "build_edge_mixed_set_main",
    "build_edge_curriculum_set_main",
    "train_edge_tracknet_main",
    "run_edge_experiment_main",
    "run_edge_replacement_search_main",
    "evaluate_phase1_main",
    "build_phase2_dataset_main",
    "train_phase2_sweep_main",
    "evaluate_phase2_main",
    "generate_phase2_report_main",
    "train_canonical_voxel_ae_main",
    "render_canonical_voxel_gallery_main",
    "render_canonical_latent_pairs_main",
    "train_baseline_main",
]

from __future__ import annotations


def test_canonical_package_apis_import_directly() -> None:
    from particle_classification.commands.data import extract_raw_zip
    from particle_classification.data.archive import pack_raw_tree, unpack_raw_archive
    from particle_classification.dbscan import CANONICAL_BACKEND, CANONICAL_PARAMS, cluster_canonical
    from particle_classification.experiments.autoencoders.path import PoseSeparatedPathAutoencoder
    from particle_classification.experiments.baseline import XYInvariantParticleNet
    from particle_classification.experiments.neural_separator.model import EdgeTrackNetTiny

    assert CANONICAL_BACKEND == "native-grid-dbscan"
    assert CANONICAL_PARAMS.eps == 5.0
    assert callable(cluster_canonical)
    assert callable(extract_raw_zip)
    assert callable(pack_raw_tree)
    assert callable(unpack_raw_archive)
    assert XYInvariantParticleNet.__name__ == "XYInvariantParticleNet"
    assert EdgeTrackNetTiny.__name__ == "EdgeTrackNetTiny"
    assert PoseSeparatedPathAutoencoder.__name__ == "PoseSeparatedPathAutoencoder"


def test_historical_phase2_facades_keep_former_public_helpers() -> None:
    import particle_classification.phase2 as point_facade
    import particle_classification.phase2_canonical_voxel_ae as canonical_voxel_facade
    import particle_classification.phase2_path_ae as path_facade
    import particle_classification.phase2_voxel_ae as voxel_facade

    for name in (
        "ParticleEncoder",
        "cluster_embeddings",
        "evaluate_phase2_model",
        "render_phase2_report",
    ):
        assert hasattr(point_facade, name)
    for name in (
        "PathArrayDataset",
        "UpsampleConvBlock",
        "augment_pose_training_batch",
        "augment_transform_path",
        "evaluate_path_ae",
        "evaluate_pose_path_ae",
        "evaluate_structured_transform_ae",
        "load_pose_split",
        "load_transform_split",
        "pd_read_csv",
        "random_pose_batch",
        "random_transform_batch",
        "relative_l2",
        "rotate_pose_vector",
        "transform_bucket_indices",
        "weighted_theta_xy",
    ):
        assert hasattr(path_facade, name)
    for name in (
        "empty_voxel_metadata",
        "ensure_voxel_channel",
        "init_voxel_weights",
    ):
        assert hasattr(voxel_facade, name)
    for name in (
        "dilated_support",
        "evaluate_canonical_model",
        "evaluate_energy_l2_model",
        "hit_count_bucket",
        "normalized_coordinate_grid",
        "random_augmentation_transform",
        "render_canonical_latent_pairs_main",
        "render_canonical_voxel_gallery_main",
        "train_canonical_voxel_ae_main",
        "transform_raw_to_params",
        "write_hard_mine_selection",
        "write_hardmine_l2_training_report",
    ):
        assert hasattr(canonical_voxel_facade, name)

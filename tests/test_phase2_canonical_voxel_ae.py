from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

from particle_classification.phase2_canonical_voxel_ae import (
    BucketBalancedVoxelSampler,
    CANONICAL_TRANSFORM_NAMES,
    HardMineL2Config,
    CanonicalTrainConfig,
    CanonicalTransformVoxelAE,
    CanonicalVoxelAEConfig,
    canonical_loss,
    energy_weighted_l2_loss,
    hardmine_reconstruction_loss,
    per_particle_energy_weighted_l2,
    relative_tensor_l2_loss,
    scan_energy_l2_losses,
    select_top_loss_indices,
    transform_density,
    transform_energy_volume,
    train_hard_mined_l2_canonical_voxel_ae_run,
    train_canonical_voxel_ae_run,
    volume_moments,
)
from particle_classification.phase2_voxel_ae import VoxelAEStage, VoxelGridConfig, VoxelSparseCache


def _tiny_model_config() -> CanonicalVoxelAEConfig:
    return CanonicalVoxelAEConfig(
        t_bins=4,
        y_bins=8,
        x_bins=8,
        patch_t=2,
        patch_y=4,
        patch_x=4,
        patch_hidden_dim=16,
        patch_embed_dim=8,
        hidden_dim=32,
        shape_latent_dim=8,
        dropout=0.0,
    )


def test_transform_identity_preserves_density_and_energy():
    density = torch.zeros((2, 4, 8, 8), dtype=torch.float32)
    density[0, 1:3, 3:5, 3:5] = 1.0
    density[1, 2, 1:7, 2:6] = 1.0
    density = density / density.sum(dim=(1, 2, 3), keepdim=True)
    transform = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0],
        ],
        dtype=torch.float32,
    )

    transformed = transform_density(density, transform, inverse=True, renormalize=True)

    assert transformed.shape == density.shape
    assert torch.allclose(transformed, density, atol=1e-6)
    assert torch.allclose(transformed.sum(dim=(1, 2, 3)), torch.ones(2), atol=1e-6)


def test_transform_layer_accepts_half_precision_inputs():
    density = torch.zeros((1, 4, 8, 8), dtype=torch.float16)
    density[0, 1:3, 3:5, 3:5] = 1.0
    density = density / density.sum()
    transform = torch.tensor([[0.0, 0.0, 0.0, 0.1, 0.05, 1.0, 1.0]], dtype=torch.float16)

    transformed = transform_density(density, transform, inverse=True, renormalize=True)

    assert transformed.dtype == torch.float32
    assert transformed.shape == density.shape
    assert torch.isfinite(transformed).all()
    assert torch.allclose(transformed.sum(dim=(1, 2, 3)), torch.ones(1), atol=1e-5)


def test_pca_transform_targets_track_xy_rotation_and_time_tilt():
    config = CanonicalVoxelAEConfig(t_bins=8, y_bins=16, x_bins=16)
    x_line = torch.zeros((1, 8, 16, 16), dtype=torch.float32)
    y_line = torch.zeros_like(x_line)
    time_tilt = torch.zeros_like(x_line)
    x_line[0, 4, 8, 4:12] = 1.0
    y_line[0, 4, 4:12, 8] = 1.0
    for idx, x in enumerate(range(4, 12)):
        time_tilt[0, idx, 8, x] = 1.0

    theta_x = volume_moments(x_line, config)["target_transform"][0, 3]
    theta_y = volume_moments(y_line, config)["target_transform"][0, 3]
    theta_time = volume_moments(time_tilt, config)["target_transform"][0, 4]

    assert abs(float(theta_x)) < 1e-5
    assert math.isclose(abs(float(theta_y)), math.pi / 2.0, rel_tol=0.0, abs_tol=1e-5)
    assert float(theta_time) > 0.5


def test_inverse_canonical_target_is_stable_under_known_xy_rotation():
    config = CanonicalVoxelAEConfig(t_bins=8, y_bins=16, x_bins=16)
    base = torch.zeros((1, 8, 16, 16), dtype=torch.float32)
    base[0, 4, 8, 4:12] = 1.0
    base = base / base.sum()
    rotate_90 = torch.tensor([[0.0, 0.0, 0.0, math.pi / 2.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    rotated = transform_energy_volume(base, rotate_90, inverse=True)

    base_transform = volume_moments(base, config)["target_transform"]
    rotated_transform = volume_moments(rotated, config)["target_transform"]
    base_canonical = transform_density(base, base_transform, inverse=False)
    rotated_canonical = transform_density(rotated, rotated_transform, inverse=False)

    assert math.isclose(abs(float(rotated_transform[0, 3] - base_transform[0, 3])), math.pi / 2.0, abs_tol=1e-5)
    assert torch.allclose(base_canonical, rotated_canonical, atol=1e-5)


def test_model_contract_decoder_uses_shape_only_and_forward_is_finite():
    config = _tiny_model_config()
    model = CanonicalTransformVoxelAE(config)
    batch = torch.zeros((3, 4, 8, 8), dtype=torch.float32)
    batch[:, 1:3, 3:5, 3:5] = 0.5

    output = model(batch)

    assert model.decoder_global[0].in_features == config.shape_latent_dim
    assert output["z_shape"].shape == (3, 8)
    assert output["z_transform"].shape == (3, len(CANONICAL_TRANSFORM_NAMES))
    assert output["canonical_density"].shape == batch.shape
    assert output["reconstruction"].shape == batch.shape
    assert torch.isfinite(output["reconstruction"]).all()
    assert torch.all(output["reconstruction"] >= 0)
    assert torch.allclose(output["canonical_density"].sum(dim=(1, 2, 3)), torch.ones(3), atol=1e-5)


def test_canonical_loss_and_shape_invariance_are_finite():
    config = _tiny_model_config()
    train_config = CanonicalTrainConfig(
        batch_size=2,
        max_steps_per_stage=1,
        min_steps_per_stage=1,
        eval_interval=1,
        max_eval_batches=1,
        stages=(VoxelAEStage("tiny", steps=1, blur_kernel=1, noise_std=0.0, voxel_dropout=0.0),),
    )
    model = CanonicalTransformVoxelAE(config)
    target = torch.zeros((2, 4, 8, 8), dtype=torch.float32)
    target[:, 1:3, 3:5, 3:5] = 0.5
    output = model(target)
    augmented_output = model(target.roll(shifts=1, dims=-1))

    loss, metrics = canonical_loss(output, target, config, train_config, augmented_output=augmented_output)

    assert torch.isfinite(loss)
    assert np.isfinite(metrics["shape_invariance_mse"])
    assert metrics["energy_relative_l1"] >= 0.0
    assert metrics["final_l1"] > 0.0
    assert metrics["final_l2_relative"] > 0.0
    assert 0.0 <= metrics["mass_overlap"] <= 1.0


def test_energy_weighted_l2_is_finite_and_penalizes_leakage():
    target = torch.zeros((2, 4, 8, 8), dtype=torch.float32)
    target[:, 1, 3:5, 3:5] = 1.0
    exact = target.clone()
    leaked = target.clone()
    leaked[:, 3, 7, 7] = 1.0
    shifted = target.roll(shifts=1, dims=-1)

    exact_loss = per_particle_energy_weighted_l2(exact, target)
    leaked_loss = per_particle_energy_weighted_l2(leaked, target)
    shifted_loss = per_particle_energy_weighted_l2(shifted, target)

    assert torch.isfinite(exact_loss).all()
    assert torch.allclose(exact_loss, torch.zeros_like(exact_loss), atol=1e-7)
    assert torch.all(leaked_loss > exact_loss)
    assert torch.all(shifted_loss > leaked_loss)


def test_energy_weighted_l2_loss_uses_final_reconstruction_and_energy_terms():
    hardmine_config = HardMineL2Config()
    target = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    target[:, 1, 3:5, 3:5] = 1.0
    output = {
        "reconstruction": target.clone(),
        "canonical_density": torch.full_like(target, 1.0 / target[0].numel()),
        "z_shape": torch.ones((1, 8)),
        "z_transform": torch.ones((1, 7)),
    }

    loss, metrics = energy_weighted_l2_loss(output, target, hardmine_config)

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)
    assert metrics["energy_weighted_l2"] == 0.0
    assert metrics["voxel_energy_weighted_l2"] == 0.0

    leaked_output = {**output, "reconstruction": target.clone()}
    leaked_output["reconstruction"][:, 3, 7, 7] = 100.0
    leaked_loss, leaked_metrics = energy_weighted_l2_loss(leaked_output, target, hardmine_config)

    assert leaked_loss > 10.0
    assert leaked_metrics["energy_relative_l1"] > 10.0
    assert leaked_metrics["support_leakage"] > 10.0


def test_relative_tensor_l2_loss_penalizes_energy_scaling_directly():
    target = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    target[:, 1, 3:5, 3:5] = 1.0
    output = {"reconstruction": target * 2.0}

    loss, metrics = relative_tensor_l2_loss(output, target)

    assert torch.isclose(loss, torch.ones_like(loss))
    assert metrics["relative_tensor_l2"] == 1.0
    dispatched, _ = hardmine_reconstruction_loss(output, target, HardMineL2Config(loss_mode="relative-l2"))
    assert torch.isclose(dispatched, loss)


def test_select_top_loss_indices_is_descending_and_fractional():
    losses = np.asarray([0.1, 0.9, 0.3, 2.0, 1.5], dtype=np.float32)

    selected = select_top_loss_indices(losses, 0.4)

    assert selected.tolist() == [3, 4]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_tiny_cache(root: Path) -> None:
    chunks = root / "chunks"
    chunks.mkdir(parents=True)
    grid = VoxelGridConfig(t_bins=4, y_bins=8, x_bins=8)
    splits = ["train"] * 4 + ["val"] * 4 + ["test"] * 4
    hit_counts = [2, 7, 20, 60] * 3
    offsets = [0]
    index_parts = []
    value_parts = []
    rows: list[dict[str, object]] = []
    for idx, (split, n_hits) in enumerate(zip(splits, hit_counts, strict=True)):
        t = idx % grid.t_bins
        y = 2 + (idx % 3)
        xs = np.arange(2, 2 + min(5, grid.x_bins - 2), dtype=np.int64)
        flat = t * grid.y_bins * grid.x_bins + y * grid.x_bins + xs
        values = np.full(xs.shape, 1.0 / len(xs), dtype=np.float32)
        index_parts.append(flat.astype(np.int32))
        value_parts.append(values)
        offsets.append(offsets[-1] + int(flat.shape[0]))
        rows.append(
            {
                "status": "ok",
                "split": split,
                "split_group": f"{split}:{idx}",
                "chunk_path": (chunks / "voxel_chunk_000000.npz").as_posix(),
                "chunk_row": idx,
                "source_npz": "tiny.particles.npz",
                "source_path": f"tiny_{split}_{idx}.t3pa",
                "particle_id": idx,
                "particle_index": idx,
                "n_hits": n_hits,
                "kept_hits": n_hits,
                "kept_fraction": 1.0,
                "kept_energy_fraction": 1.0,
                "occupied_voxels": int(flat.shape[0]),
                "x_span": int(flat.shape[0]),
                "y_span": 1,
                "time_span": 1,
                "schema_version": "test",
            }
        )
    np.savez_compressed(
        chunks / "voxel_chunk_000000.npz",
        voxel_offsets=np.asarray(offsets, dtype=np.int64),
        voxel_index=np.concatenate(index_parts).astype(np.int32),
        voxel_value=np.concatenate(value_parts).astype(np.float32),
    )
    _write_rows(root / "manifest.csv", rows)


def test_bucket_balanced_sampler_emits_all_available_buckets(tmp_path):
    _make_tiny_cache(tmp_path)
    dataset = VoxelSparseCache(tmp_path, split="train", grid_config=VoxelGridConfig(t_bins=4, y_bins=8, x_bins=8))
    sampler = BucketBalancedVoxelSampler(dataset, seed=17)

    rows = sampler.sample_rows(8)

    buckets = set()
    for row in rows:
        n_hits = int(float(row["n_hits"]))
        if n_hits <= 4:
            buckets.add("2-4")
        elif n_hits <= 10:
            buckets.add("5-10")
        elif n_hits <= 50:
            buckets.add("11-50")
        else:
            buckets.add("51+")
    assert buckets == {"2-4", "5-10", "11-50", "51+"}


def test_scan_energy_l2_losses_is_deterministic_and_train_only(tmp_path):
    _make_tiny_cache(tmp_path)
    grid_config = VoxelGridConfig(t_bins=4, y_bins=8, x_bins=8)
    dataset = VoxelSparseCache(tmp_path, split="train", grid_config=grid_config)
    model = CanonicalTransformVoxelAE(_tiny_model_config())
    stage = VoxelAEStage("highest", steps=1, blur_kernel=5, blur_mix=1.0, noise_std=0.04, voxel_dropout=0.08)
    hardmine_config = HardMineL2Config()

    first = scan_energy_l2_losses(
        model,
        dataset,
        stage,
        hardmine_config,
        batch_size=2,
        device="cpu",
        seed=123,
        mine_clean_loss=True,
    )
    second = scan_energy_l2_losses(
        model,
        dataset,
        stage,
        hardmine_config,
        batch_size=2,
        device="cpu",
        seed=123,
        mine_clean_loss=True,
    )

    assert len(first["row_indices"]) == 4
    assert all(row["split"] == "train" for row in dataset.rows)
    assert np.allclose(first["corrupted_l2"], second["corrupted_l2"])
    assert np.allclose(first["clean_l2"], second["clean_l2"])


def test_tiny_train_run_writes_checkpoint_and_resets_lr_per_stage(tmp_path):
    cache = tmp_path / "cache"
    run = tmp_path / "run"
    _make_tiny_cache(cache)
    grid_config = VoxelGridConfig(t_bins=4, y_bins=8, x_bins=8)
    model_config = _tiny_model_config()
    train_config = CanonicalTrainConfig(
        batch_size=2,
        learning_rate=1e-4,
        max_steps_per_stage=1,
        min_steps_per_stage=1,
        eval_interval=1,
        max_eval_batches=1,
        amp=False,
        stages=(
            VoxelAEStage("tiny_level_1", steps=1, blur_kernel=1, noise_std=0.0, voxel_dropout=0.0),
            VoxelAEStage("tiny_level_2", steps=1, blur_kernel=1, noise_std=0.0, voxel_dropout=0.0),
        ),
    )

    summary = train_canonical_voxel_ae_run(
        cache,
        run,
        model_config=model_config,
        grid_config=grid_config,
        train_config=train_config,
        device="cpu",
        verbose=False,
    )

    assert summary["model_type"] == "canonical_transform_voxel_ae"
    assert (run / "checkpoint.pt").exists()
    with (run / "stage_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        stage_rows = list(csv.DictReader(handle))
    assert len(stage_rows) == 2
    assert all(float(row["initial_lr"]) == 1e-4 for row in stage_rows)


def test_tiny_hardmine_l2_train_run_writes_mining_artifacts_and_fixed_corruption(tmp_path):
    cache = tmp_path / "cache"
    run = tmp_path / "hardmine_run"
    init = tmp_path / "init.pt"
    _make_tiny_cache(cache)
    grid_config = VoxelGridConfig(t_bins=4, y_bins=8, x_bins=8)
    model_config = _tiny_model_config()
    torch.save(
        {
            "model_state_dict": CanonicalTransformVoxelAE(model_config).state_dict(),
            "model_config": model_config.__dict__,
            "grid_config": grid_config.__dict__,
            "model_type": "test_init",
        },
        init,
    )
    train_config = CanonicalTrainConfig(
        batch_size=2,
        learning_rate=1e-4,
        max_steps_per_stage=1,
        min_steps_per_stage=1,
        eval_interval=1,
        max_eval_batches=1,
        amp=False,
        stages=(VoxelAEStage("unused", steps=1, blur_kernel=1, noise_std=0.0, voxel_dropout=0.0),),
    )
    hardmine_config = HardMineL2Config(
        phases=1,
        fraction=0.5,
        scan_batch_size=2,
        smoke_steps=1,
        smoke_scan_items=4,
        mine_clean_loss=True,
        smoke_compare_init=True,
    )

    summary = train_hard_mined_l2_canonical_voxel_ae_run(
        cache,
        run,
        model_config=model_config,
        grid_config=grid_config,
        train_config=train_config,
        hardmine_config=hardmine_config,
        device="cpu",
        init_checkpoint=init,
        verbose=False,
    )

    assert summary["loss_mode"] == "energy-weighted-l2"
    assert (run / "checkpoint.pt").exists()
    assert (run / "mining" / "phase_01_selected.csv").exists()
    with (run / "mining_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        mining_rows = list(csv.DictReader(handle))
    assert mining_rows[0]["rows_scanned"] == "4"
    assert mining_rows[0]["selected_rows"] == "2"
    with (run / "phase_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        phase_rows = list(csv.DictReader(handle))
    assert phase_rows[0]["blur_kernel"] == "5"
    assert phase_rows[0]["noise_std"] == "0.04"
    assert phase_rows[0]["voxel_dropout"] == "0.08"

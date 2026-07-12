from __future__ import annotations

import numpy as np
import torch

from particle_classification.phase2_voxel_ae import (
    VoxelAETrainConfig,
    VoxelGridConfig,
    VoxelMLPConfig,
    VoxelPatchMLPAutoencoder,
    VoxelSparseCache,
    build_voxel_energy_cache,
    corrupt_voxel_batch,
    voxel_ae_loss,
    voxelize_particle_energy,
)


def test_voxelize_particle_energy_centers_sums_and_crops():
    config = VoxelGridConfig(t_bins=4, y_bins=4, x_bins=10, time_bin=1.0)
    x = np.asarray([10, 10, 11, 20], dtype=np.float64)
    y = np.asarray([20, 20, 21, 21], dtype=np.float64)
    t = np.asarray([5.0, 5.0, 6.0, 6.0], dtype=np.float64)
    e = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    idx, value, meta = voxelize_particle_energy(x, y, t, e, config)
    assert idx.dtype == np.int32
    assert value.dtype == np.float32
    assert meta["n_hits"] == 4
    assert meta["kept_hits"] < 4
    assert np.isclose(float(value.sum()), (1.0 + 2.0 + 3.0) / config.energy_norm)
    assert len(idx) == 2


def test_voxel_patch_mlp_forward_loss_and_corruption_are_finite():
    config = VoxelMLPConfig(
        t_bins=4,
        y_bins=8,
        x_bins=8,
        patch_t=2,
        patch_y=4,
        patch_x=4,
        patch_hidden_dim=16,
        patch_embed_dim=8,
        hidden_dim=32,
        shape_latent_dim=4,
        aux_latent_dim=3,
    )
    model = VoxelPatchMLPAutoencoder(config)
    batch = torch.zeros((2, 4, 8, 8), dtype=torch.float32)
    batch[:, 1:3, 3:5, 3:5] = 0.5
    noisy = corrupt_voxel_batch(batch, blur_kernel=3, noise_std=0.01, voxel_dropout=0.1)
    output = model(noisy)
    assert output["z_shape"].shape == (2, 4)
    assert output["z_aux"].shape == (2, 3)
    assert output["reconstruction"].shape == batch.shape
    loss, metrics = voxel_ae_loss(output["reconstruction"], batch)
    assert torch.isfinite(loss)
    assert np.isfinite(metrics["mse"])
    assert VoxelAETrainConfig().total_steps > 0


def write_tiny_particle_npz(path):
    hit_x = np.asarray([10, 11, 12, 50, 51], dtype=np.uint16)
    hit_y = np.asarray([20, 21, 22, 60, 61], dtype=np.uint16)
    hit_time = np.asarray([0.0, 0.625, 1.25, 10.0, 10.625], dtype=np.float64)
    hit_energy = np.asarray([1.0, 1.2, 1.1, 0.7, 0.8], dtype=np.float32)
    np.savez_compressed(
        path,
        hit_x=hit_x,
        hit_y=hit_y,
        hit_time=hit_time,
        hit_energy=hit_energy,
        particle_offsets=np.asarray([0, 3, 5], dtype=np.int64),
        particle_id=np.asarray([7, 8], dtype=np.int32),
        particle_energy_sum=np.asarray([3.3, 1.5], dtype=np.float32),
        source_path=np.asarray("tiny.t3pa"),
    )


def test_voxel_sparse_cache_builds_and_densifies(tmp_path):
    particles = tmp_path / "particles"
    particles.mkdir()
    write_tiny_particle_npz(particles / "tiny.particles.npz")
    cache = tmp_path / "cache"
    grid = VoxelGridConfig(t_bins=8, y_bins=8, x_bins=8, time_bin=0.625, max_particles=2, chunk_size=1)
    summary = build_voxel_energy_cache(particles, cache, config=grid)
    assert summary["particles"] == 2
    assert (cache / "manifest.csv").exists()
    rows = []
    for split in ["train", "val", "test"]:
        ds = VoxelSparseCache(cache, split=split, grid_config=grid)
        rows.extend(ds.rows)
        if len(ds):
            batch = ds.random_batch(1, "cpu")
            assert batch.shape == (1, 8, 8, 8)
            assert float(batch.sum()) > 0
    assert len(rows) == 2

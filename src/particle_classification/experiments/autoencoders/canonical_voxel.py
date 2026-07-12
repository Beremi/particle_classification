from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...dbscan.pipeline import write_dict_rows
from .voxel import (
    VoxelAEStage,
    VoxelGridConfig,
    VoxelSparseCache,
    corrupt_voxel_batch,
    ensure_voxel_channel,
    init_voxel_weights,
    make_decay_voxel_curriculum_stages,
)


CANONICAL_TRANSFORM_NAMES = (
    "dx_norm",
    "dy_norm",
    "dt_norm",
    "theta_xy",
    "theta_time_tilt",
    "scale_xyz",
    "energy_scale",
)


@dataclass(frozen=True)
class CanonicalVoxelAEConfig:
    t_bins: int = 32
    y_bins: int = 64
    x_bins: int = 64
    patch_t: int = 4
    patch_y: int = 8
    patch_x: int = 8
    patch_hidden_dim: int = 192
    patch_embed_dim: int = 32
    hidden_dim: int = 768
    shape_latent_dim: int = 8
    transform_dim: int = 7
    transform_mode: str = "full"
    encoder_hidden_layers: int = 1
    decoder_hidden_layers: int = 0
    dropout: float = 0.03
    max_shift_norm: float = 0.45
    max_tilt_rad: float = math.pi / 3.0
    min_scale: float = 0.35
    max_scale: float = 2.75
    canonical_rms_norm: float = 0.23
    output_temperature: float = 1.0

    @property
    def flat_dim(self) -> int:
        return self.t_bins * self.y_bins * self.x_bins


@dataclass(frozen=True)
class CanonicalTrainConfig:
    seed: int = 20260507
    batch_size: int = 128
    learning_rate: float = 1e-4
    lr_decay: float = 0.5
    lr_patience_evals: int = 4
    stop_lr_below: float = 5e-6
    min_steps_per_stage: int = 4096
    max_steps_per_stage: int = 20512
    eval_interval: int = 512
    max_eval_batches: int = 12
    weight_decay: float = 2e-4
    grad_clip: float = 1.0
    amp: bool = True
    final_l1_weight: float = 1.0
    final_mse_weight: float = 0.25
    canonical_l1_weight: float = 0.35
    transform_weight: float = 0.12
    energy_weight: float = 0.35
    support_weight: float = 0.08
    invariance_weight: float = 0.04
    augmentation_weight: float = 0.35
    early_stop_min_delta: float = 1e-5
    wall_time_limit_s: float = 12 * 60 * 60
    stages: tuple[VoxelAEStage, ...] = make_decay_voxel_curriculum_stages(
        phases=10,
        steps_per_phase=20512,
        start_blur_kernel=5,
        start_blur_mix=1.0,
        start_noise_std=0.04,
        start_voxel_dropout=0.08,
        final_fraction=0.1,
    )


@dataclass(frozen=True)
class HardMineL2Config:
    phases: int = 10
    fraction: float = 0.10
    scan_batch_size: int = 512
    smoke_steps: int = 256
    smoke_scan_items: int = 20_000
    smoke_eval_batches: int = 4
    warm_start_prefer_margin: float = 0.03
    background_weight: float = 1.0
    support_weight: float = 1.0
    energy_weight: float = 4.0
    energy_total_weight: float = 1.0
    outside_energy_weight: float = 0.25
    loss_mode: str = "energy-weighted-l2"
    mine_clean_loss: bool = False
    smoke_compare_init: bool = False
    fixed_stage: VoxelAEStage = VoxelAEStage(
        "hardmine_highest_corruption",
        steps=20512,
        blur_kernel=5,
        blur_mix=1.0,
        noise_std=0.04,
        voxel_dropout=0.08,
    )


def _axis_grid_1d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)


def normalized_coordinate_grid(
    batch_size: int,
    t_bins: int,
    y_bins: int,
    x_bins: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    z, y, x = torch.meshgrid(
        _axis_grid_1d(t_bins, device, dtype),
        _axis_grid_1d(y_bins, device, dtype),
        _axis_grid_1d(x_bins, device, dtype),
        indexing="ij",
    )
    grid = torch.stack([x, y, z], dim=-1)
    return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)


def _rotation_matrices(theta_xy: torch.Tensor, theta_time: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    b = theta_xy.shape[0]
    dtype = theta_xy.dtype
    device = theta_xy.device
    cz = torch.cos(theta_xy)
    sz = torch.sin(theta_xy)
    cy = torch.cos(theta_time)
    sy = torch.sin(theta_time)
    zeros = torch.zeros_like(cz)
    ones = torch.ones_like(cz)
    r_xy = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=1,
    )
    r_tilt = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=1,
    )
    scale_m = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(b, 1, 1) * scale.view(b, 1, 1)
    return r_xy @ r_tilt @ scale_m


def affine_grid_from_transform(
    transform: torch.Tensor,
    output_shape: tuple[int, int, int, int, int],
    *,
    inverse: bool = True,
) -> torch.Tensor:
    """Build a 3D sampling grid for ``grid_sample``.

    Transform convention: ``p_out = A p_canonical + translation`` in normalized
    ``x,y,t`` coordinates. ``inverse=True`` samples a canonical source into the
    transformed output space. ``inverse=False`` samples a transformed source
    into canonical output space.
    """

    b = transform.shape[0]
    _, _, t_bins, y_bins, x_bins = output_shape
    grid = normalized_coordinate_grid(b, t_bins, y_bins, x_bins, device=transform.device, dtype=transform.dtype)
    translation = transform[:, 0:3]
    theta_xy = transform[:, 3]
    theta_time = transform[:, 4]
    scale = transform[:, 5].clamp_min(1e-4)
    a = _rotation_matrices(theta_xy, theta_time, scale)
    if inverse:
        a_map = torch.linalg.inv(a)
        shifted = grid.reshape(b, -1, 3) - translation[:, None, :]
        mapped = torch.bmm(shifted, a_map.transpose(1, 2))
    else:
        mapped = torch.bmm(grid.reshape(b, -1, 3), a.transpose(1, 2)) + translation[:, None, :]
    return mapped.view(b, t_bins, y_bins, x_bins, 3)


def _normalize_density(volume: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    reduce_dims = tuple(range(1, volume.ndim))
    return volume / torch.sum(volume, dim=reduce_dims, keepdim=True).clamp_min(eps)


def transform_density(
    density: torch.Tensor,
    transform: torch.Tensor,
    *,
    inverse: bool = True,
    renormalize: bool = True,
) -> torch.Tensor:
    input_was_4d = density.ndim == 4
    device_type = density.device.type if density.device.type in {"cpu", "cuda"} else "cuda"
    with torch.amp.autocast(device_type, enabled=False):
        density_5d = ensure_voxel_channel(density).float()
        grid = affine_grid_from_transform(transform.float(), tuple(density_5d.shape), inverse=inverse)
        sampled = F.grid_sample(density_5d, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.clamp_min(0.0)
        sampled_4d = sampled[:, 0]
        if renormalize:
            sampled_4d = _normalize_density(sampled_4d)
    return sampled_4d if input_was_4d else sampled_4d.unsqueeze(1)


def transform_energy_volume(volume: torch.Tensor, transform: torch.Tensor, *, inverse: bool = True) -> torch.Tensor:
    input_was_4d = volume.ndim == 4
    volume_4d = ensure_voxel_channel(volume)[:, 0]
    energy = torch.sum(volume_4d, dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    density = volume_4d / energy
    transformed = transform_density(density, transform, inverse=inverse, renormalize=True) * energy
    return transformed if input_was_4d else transformed.unsqueeze(1)


def volume_moments(volume: torch.Tensor, config: CanonicalVoxelAEConfig) -> dict[str, torch.Tensor]:
    volume = ensure_voxel_channel(volume)[:, 0].clamp_min(0.0)
    b, t_bins, y_bins, x_bins = volume.shape
    dtype = volume.dtype
    device = volume.device
    grid = normalized_coordinate_grid(b, t_bins, y_bins, x_bins, device=device, dtype=dtype)
    weights = volume.reshape(b, -1)
    energy = weights.sum(dim=1).clamp_min(1e-8)
    coords = grid.reshape(b, -1, 3)
    center = torch.sum(coords * weights[:, :, None], dim=1) / energy[:, None]
    centered = coords - center[:, None, :]
    cov = torch.bmm((centered * weights[:, :, None]).transpose(1, 2), centered) / energy[:, None, None]
    cov_xy = cov[:, :2, :2]
    evals_xy, evecs_xy = torch.linalg.eigh(cov_xy)
    major_xy = evecs_xy[:, :, -1]
    theta_xy = torch.atan2(major_xy[:, 1], major_xy[:, 0])
    theta_xy = torch.remainder(theta_xy + math.pi / 2.0, math.pi) - math.pi / 2.0

    cos_t = torch.cos(theta_xy)
    sin_t = torch.sin(theta_xy)
    u = centered[:, :, 0] * cos_t[:, None] + centered[:, :, 1] * sin_t[:, None]
    z = centered[:, :, 2]
    m_uu = torch.sum(weights * u * u, dim=1) / energy
    m_zz = torch.sum(weights * z * z, dim=1) / energy
    m_uz = torch.sum(weights * u * z, dim=1) / energy
    cov_ut = torch.stack(
        [
            torch.stack([m_uu, m_uz], dim=-1),
            torch.stack([m_uz, m_zz], dim=-1),
        ],
        dim=1,
    )
    _, evecs_ut = torch.linalg.eigh(cov_ut)
    major_ut = evecs_ut[:, :, -1]
    sign = torch.where(major_ut[:, 0:1] < 0, -1.0, 1.0)
    major_ut = major_ut * sign
    theta_time = torch.atan2(major_ut[:, 1], major_ut[:, 0]).clamp(-config.max_tilt_rad, config.max_tilt_rad)
    rms = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2).sum(dim=1).clamp_min(1e-8))
    scale = (rms / config.canonical_rms_norm).clamp(config.min_scale, config.max_scale)
    target = torch.stack(
        [
            center[:, 0].clamp(-config.max_shift_norm, config.max_shift_norm),
            center[:, 1].clamp(-config.max_shift_norm, config.max_shift_norm),
            center[:, 2].clamp(-config.max_shift_norm, config.max_shift_norm),
            theta_xy,
            theta_time,
            scale,
            energy,
        ],
        dim=1,
    )
    evals_xy = evals_xy.clamp_min(0.0)
    linearity_xy = 1.0 - evals_xy[:, 0] / evals_xy[:, 1].clamp_min(1e-8)
    return {
        "target_transform": target,
        "energy": energy,
        "linearity_xy": linearity_xy,
        "center": center,
    }


def transform_raw_to_params(raw: torch.Tensor, config: CanonicalVoxelAEConfig) -> torch.Tensor:
    if config.transform_mode == "xy_energy":
        shift = raw.new_zeros((raw.shape[0], 3))
        theta_xy = torch.tanh(raw[:, 0:1]) * math.pi
        theta_time = raw.new_zeros((raw.shape[0], 1))
        scale = raw.new_ones((raw.shape[0], 1))
        energy = F.softplus(raw[:, 1:2]).clamp_min(1e-8)
        return torch.cat([shift, theta_xy, theta_time, scale, energy], dim=1)
    if config.transform_mode != "full":
        raise ValueError(f"Unknown transform_mode: {config.transform_mode}")
    shift = torch.tanh(raw[:, 0:3]) * config.max_shift_norm
    theta_xy = torch.tanh(raw[:, 3:4]) * math.pi
    theta_time = torch.tanh(raw[:, 4:5]) * config.max_tilt_rad
    scale = config.min_scale + (config.max_scale - config.min_scale) * torch.sigmoid(raw[:, 5:6])
    energy = F.softplus(raw[:, 6:7]).clamp_min(1e-8)
    return torch.cat([shift, theta_xy, theta_time, scale, energy], dim=1)


class CanonicalTransformVoxelAE(nn.Module):
    def __init__(self, config: CanonicalVoxelAEConfig):
        super().__init__()
        self.config = config
        if config.t_bins % config.patch_t or config.y_bins % config.patch_y or config.x_bins % config.patch_x:
            raise ValueError("Voxel dimensions must be divisible by patch dimensions")
        patch_dim = config.patch_t * config.patch_y * config.patch_x
        n_patches = (config.t_bins // config.patch_t) * (config.y_bins // config.patch_y) * (config.x_bins // config.patch_x)
        self.patch_dim = patch_dim
        self.n_patches = n_patches
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_dim, config.patch_hidden_dim),
            nn.LayerNorm(config.patch_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.patch_hidden_dim, config.patch_embed_dim),
            nn.SiLU(),
        )
        encoder_layers: list[nn.Module] = [
            nn.Linear(n_patches * config.patch_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(config.encoder_hidden_layers):
            encoder_layers.extend(
                [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.SiLU(),
                ]
            )
        self.encoder = nn.Sequential(*encoder_layers)
        self.shape_head = nn.Linear(config.hidden_dim, config.shape_latent_dim)
        transform_head_dim = 2 if config.transform_mode == "xy_energy" else config.transform_dim
        self.transform_head = nn.Linear(config.hidden_dim, transform_head_dim)
        decoder_layers: list[nn.Module] = [
            nn.Linear(config.shape_latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(config.decoder_hidden_layers):
            decoder_layers.extend(
                [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.SiLU(),
                ]
            )
        decoder_layers.extend([nn.Linear(config.hidden_dim, n_patches * config.patch_embed_dim), nn.SiLU()])
        self.decoder_global = nn.Sequential(*decoder_layers)
        self.patch_decoder = nn.Sequential(
            nn.Linear(config.patch_embed_dim, config.patch_hidden_dim),
            nn.LayerNorm(config.patch_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.patch_hidden_dim, patch_dim),
        )
        self.apply(init_voxel_weights)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_voxel_channel(x)
        cfg = self.config
        b, c, t, y, xdim = x.shape
        patches = x.reshape(
            b,
            c,
            t // cfg.patch_t,
            cfg.patch_t,
            y // cfg.patch_y,
            cfg.patch_y,
            xdim // cfg.patch_x,
            cfg.patch_x,
        )
        patches = patches.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return patches.view(b, self.n_patches, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        b = patches.shape[0]
        patches = patches.view(
            b,
            cfg.t_bins // cfg.patch_t,
            cfg.y_bins // cfg.patch_y,
            cfg.x_bins // cfg.patch_x,
            1,
            cfg.patch_t,
            cfg.patch_y,
            cfg.patch_x,
        )
        volume = patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return volume.view(b, 1, cfg.t_bins, cfg.y_bins, cfg.x_bins)

    def encode(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.patch_encoder(self.patchify(volume))
        hidden = self.encoder(patches.reshape(patches.shape[0], -1))
        z_shape = self.shape_head(hidden)
        z_transform = transform_raw_to_params(self.transform_head(hidden), self.config)
        return z_shape, z_transform

    def decode_canonical(self, z_shape: torch.Tensor) -> torch.Tensor:
        patch_embed = self.decoder_global(z_shape).view(z_shape.shape[0], self.n_patches, self.config.patch_embed_dim)
        patches = self.patch_decoder(patch_embed)
        raw = self.unpatchify(patches)[:, 0]
        density = F.softplus(raw / max(self.config.output_temperature, 1e-6))
        return _normalize_density(density)

    def forward(self, volume: torch.Tensor) -> dict[str, torch.Tensor]:
        input_was_4d = volume.ndim == 4
        z_shape, z_transform = self.encode(volume)
        canonical_density = self.decode_canonical(z_shape)
        transformed_density = transform_density(canonical_density, z_transform, inverse=True, renormalize=True)
        reconstruction = transformed_density * z_transform[:, 6].view(-1, 1, 1, 1)
        if not input_was_4d:
            reconstruction = reconstruction.unsqueeze(1)
            canonical_density = canonical_density.unsqueeze(1)
            transformed_density = transformed_density.unsqueeze(1)
        return {
            "z_shape": z_shape,
            "z_transform": z_transform,
            "canonical_density": canonical_density,
            "transformed_density": transformed_density,
            "reconstruction": reconstruction,
        }


class BucketBalancedVoxelSampler:
    def __init__(self, dataset: VoxelSparseCache, *, seed: int = 20260507):
        self.dataset = dataset
        self.rng = random.Random(f"{seed}:bucket-balanced:{dataset.cache_dir}:{len(dataset.rows)}")
        self.buckets: dict[str, list[dict[str, str]]] = {"2-4": [], "5-10": [], "11-50": [], "51+": []}
        for row in dataset.rows:
            n_hits = int(float(row.get("n_hits", 0)))
            if n_hits <= 4:
                self.buckets["2-4"].append(row)
            elif n_hits <= 10:
                self.buckets["5-10"].append(row)
            elif n_hits <= 50:
                self.buckets["11-50"].append(row)
            else:
                self.buckets["51+"].append(row)
        self.nonempty = [name for name, rows in self.buckets.items() if rows]
        if not self.nonempty:
            raise ValueError("No rows available for balanced sampling")

    def sample_rows(self, batch_size: int) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        base = batch_size // len(self.nonempty)
        rem = batch_size % len(self.nonempty)
        for idx, name in enumerate(self.nonempty):
            count = base + (1 if idx < rem else 0)
            bucket = self.buckets[name]
            rows.extend(bucket[self.rng.randrange(len(bucket))] for _ in range(count))
        self.rng.shuffle(rows)
        return rows

    def batch(self, batch_size: int, device: str | torch.device) -> torch.Tensor:
        return self.dataset.rows_to_dense(self.sample_rows(batch_size), device)


class RowSubsetVoxelSampler:
    def __init__(self, dataset: VoxelSparseCache, row_indices: Iterable[int], *, seed: int = 20260507):
        self.dataset = dataset
        self.row_indices = [int(index) for index in row_indices]
        if not self.row_indices:
            raise ValueError("Hard-mined row subset is empty")
        self.rng = random.Random(f"{seed}:row-subset:{dataset.cache_dir}:{len(self.row_indices)}")

    def sample_rows(self, batch_size: int) -> list[dict[str, str]]:
        return [self.dataset.rows[self.row_indices[self.rng.randrange(len(self.row_indices))]] for _ in range(batch_size)]

    def batch(self, batch_size: int, device: str | torch.device) -> torch.Tensor:
        return self.dataset.rows_to_dense(self.sample_rows(batch_size), device)


def hit_count_bucket(row: dict[str, object]) -> str:
    n_hits = int(float(row.get("n_hits", 0)))
    if n_hits <= 4:
        return "2-4"
    if n_hits <= 10:
        return "5-10"
    if n_hits <= 50:
        return "11-50"
    return "51+"


def per_particle_energy_weighted_l2(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    background_weight: float = 1.0,
    support_weight: float = 1.0,
    energy_weight: float = 4.0,
) -> torch.Tensor:
    target = ensure_voxel_channel(target)[:, 0]
    reconstruction = ensure_voxel_channel(reconstruction)[:, 0]
    diff = reconstruction - target
    reduce_dims = (1, 2, 3)
    target_max = torch.amax(target, dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    occupied = (target > 0).to(target.dtype)
    weights = float(background_weight) + float(support_weight) * occupied + float(energy_weight) * (target / target_max)
    numerator = torch.sum(weights * diff * diff, dim=reduce_dims)
    denominator = torch.sum(weights * target * target, dim=reduce_dims).clamp_min(1e-8)
    return torch.sqrt(numerator / denominator)


def energy_weighted_l2_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    hardmine_config: HardMineL2Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_4d = ensure_voxel_channel(target)[:, 0]
    reconstruction = ensure_voxel_channel(output["reconstruction"])[:, 0]
    voxel_l2 = per_particle_energy_weighted_l2(
        reconstruction,
        target_4d,
        background_weight=hardmine_config.background_weight,
        support_weight=hardmine_config.support_weight,
        energy_weight=hardmine_config.energy_weight,
    )
    target_energy = torch.sum(target_4d, dim=(1, 2, 3)).clamp_min(1e-8)
    recon_energy = torch.sum(reconstruction, dim=(1, 2, 3))
    energy_relative_per_particle = torch.abs(recon_energy - target_energy) / target_energy
    outside = ~dilated_support(target_4d)
    outside_energy_per_particle = torch.sum(torch.where(outside, reconstruction, torch.zeros_like(reconstruction)), dim=(1, 2, 3)) / target_energy
    per_particle = torch.sqrt(
        voxel_l2 * voxel_l2
        + float(hardmine_config.energy_total_weight) * energy_relative_per_particle * energy_relative_per_particle
        + float(hardmine_config.outside_energy_weight) * outside_energy_per_particle * outside_energy_per_particle
    )
    loss = per_particle.mean()
    energy_relative = torch.mean(energy_relative_per_particle)
    mass_overlap = torch.mean(torch.sum(torch.minimum(reconstruction, target_4d), dim=(1, 2, 3)) / target_energy)
    support_leakage = torch.mean(outside_energy_per_particle)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "energy_weighted_l2": float(loss.detach().cpu()),
        "voxel_energy_weighted_l2": float(voxel_l2.mean().detach().cpu()),
        "energy_weighted_l2_p50": float(torch.quantile(per_particle.detach(), 0.50).cpu()),
        "energy_weighted_l2_p95": float(torch.quantile(per_particle.detach(), 0.95).cpu()),
        "energy_relative_l1": float(energy_relative.detach().cpu()),
        "mass_overlap": float(mass_overlap.detach().cpu()),
        "support_leakage": float(support_leakage.detach().cpu()),
    }
    return loss, metrics


def relative_tensor_l2_loss(output: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    target_4d = ensure_voxel_channel(target)[:, 0]
    reconstruction = ensure_voxel_channel(output["reconstruction"])[:, 0]
    diff = reconstruction - target_4d
    reduce_dims = (1, 2, 3)
    target_l2 = torch.sqrt(torch.sum(target_4d * target_4d, dim=reduce_dims)).clamp_min(1e-8)
    per_particle = torch.sqrt(torch.sum(diff * diff, dim=reduce_dims)) / target_l2
    loss = per_particle.mean()
    target_energy = torch.sum(target_4d, dim=reduce_dims).clamp_min(1e-8)
    recon_energy = torch.sum(reconstruction, dim=reduce_dims)
    energy_relative = torch.mean(torch.abs(recon_energy - target_energy) / target_energy)
    outside = ~dilated_support(target_4d)
    support_leakage = torch.mean(torch.sum(torch.where(outside, reconstruction, torch.zeros_like(reconstruction)), dim=reduce_dims) / target_energy)
    mass_overlap = torch.mean(torch.sum(torch.minimum(reconstruction, target_4d), dim=reduce_dims) / target_energy)
    return loss, {
        "loss": float(loss.detach().cpu()),
        "relative_tensor_l2": float(loss.detach().cpu()),
        "relative_tensor_l2_p50": float(torch.quantile(per_particle.detach(), 0.50).cpu()),
        "relative_tensor_l2_p95": float(torch.quantile(per_particle.detach(), 0.95).cpu()),
        "energy_relative_l1": float(energy_relative.detach().cpu()),
        "mass_overlap": float(mass_overlap.detach().cpu()),
        "support_leakage": float(support_leakage.detach().cpu()),
    }


def hardmine_reconstruction_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    hardmine_config: HardMineL2Config,
) -> tuple[torch.Tensor, dict[str, float]]:
    if hardmine_config.loss_mode == "relative-l2":
        return relative_tensor_l2_loss(output, target)
    if hardmine_config.loss_mode == "energy-weighted-l2":
        return energy_weighted_l2_loss(output, target, hardmine_config)
    raise ValueError(f"Unsupported hardmine loss mode: {hardmine_config.loss_mode}")


def select_top_loss_indices(losses: np.ndarray, fraction: float) -> np.ndarray:
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    losses = np.asarray(losses, dtype=np.float32)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a nonempty 1D array")
    count = max(1, int(math.ceil(losses.size * fraction)))
    selected = np.argpartition(losses, -count)[-count:]
    selected = selected[np.argsort(losses[selected])[::-1]]
    return selected.astype(np.int64)


def dilated_support(target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    target_5d = ensure_voxel_channel(target)
    support = (target_5d > 0).float()
    dilated = F.max_pool3d(support, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return dilated[:, 0] > 0


def canonical_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    model_config: CanonicalVoxelAEConfig,
    train_config: CanonicalTrainConfig,
    *,
    augmented_output: dict[str, torch.Tensor] | None = None,
    target_transform: torch.Tensor | None = None,
    canonical_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = ensure_voxel_channel(target)[:, 0]
    reconstruction = ensure_voxel_channel(output["reconstruction"])[:, 0]
    target_transform = target_transform if target_transform is not None else volume_moments(target, model_config)["target_transform"]
    if canonical_target is None:
        target_density = _normalize_density(target)
        canonical_target = transform_density(target_density, target_transform, inverse=False, renormalize=True)

    target_energy = target_transform[:, 6].clamp_min(1e-6)
    reconstruction_error = reconstruction - target
    final_l1 = torch.mean(torch.sum(torch.abs(reconstruction_error), dim=(1, 2, 3)) / target_energy)
    final_mse = torch.mean(reconstruction_error**2)
    final_l2_relative = torch.mean(
        torch.sqrt(torch.sum(reconstruction_error**2, dim=(1, 2, 3)))
        / torch.sqrt(torch.sum(target**2, dim=(1, 2, 3))).clamp_min(1e-6)
    )
    canonical_l1 = torch.mean(torch.sum(torch.abs(output["canonical_density"] - canonical_target), dim=(1, 2, 3)))
    mass_overlap = torch.mean(torch.sum(torch.minimum(reconstruction, target), dim=(1, 2, 3)) / target_energy)

    z_transform = output["z_transform"]
    shift_loss = torch.mean((z_transform[:, 0:3] - target_transform[:, 0:3]) ** 2)
    theta_xy_loss = torch.mean(1.0 - torch.cos(2.0 * (z_transform[:, 3] - target_transform[:, 3])))
    theta_time_loss = torch.mean(1.0 - torch.cos(z_transform[:, 4] - target_transform[:, 4]))
    scale_loss = torch.mean(torch.abs(z_transform[:, 5] - target_transform[:, 5]) / target_transform[:, 5].clamp_min(1e-6))
    recon_energy = torch.sum(reconstruction, dim=(1, 2, 3))
    energy_relative = torch.mean(torch.abs(recon_energy - target_energy) / target_energy)
    transform_loss = shift_loss + 0.25 * theta_xy_loss + 0.25 * theta_time_loss + scale_loss + energy_relative

    outside = ~dilated_support(target)
    outside_energy = torch.sum(torch.where(outside, reconstruction, torch.zeros_like(reconstruction)), dim=(1, 2, 3))
    support_leakage = torch.mean(outside_energy / target_energy)
    invariance_loss = torch.zeros((), dtype=target.dtype, device=target.device)
    if augmented_output is not None:
        invariance_loss = torch.mean((output["z_shape"] - augmented_output["z_shape"]) ** 2)
    loss = (
        train_config.final_l1_weight * final_l1
        + train_config.final_mse_weight * final_l2_relative
        + train_config.canonical_l1_weight * canonical_l1
        + train_config.transform_weight * transform_loss
        + train_config.energy_weight * energy_relative
        + train_config.support_weight * support_leakage
        + train_config.invariance_weight * invariance_loss
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "final_l1": float(final_l1.detach().cpu()),
        "final_mse": float(final_mse.detach().cpu()),
        "final_l2_relative": float(final_l2_relative.detach().cpu()),
        "canonical_l1": float(canonical_l1.detach().cpu()),
        "mass_overlap": float(mass_overlap.detach().cpu()),
        "shift_mse": float(shift_loss.detach().cpu()),
        "theta_xy_loss": float(theta_xy_loss.detach().cpu()),
        "theta_time_loss": float(theta_time_loss.detach().cpu()),
        "scale_relative_l1": float(scale_loss.detach().cpu()),
        "energy_relative_l1": float(energy_relative.detach().cpu()),
        "support_leakage": float(support_leakage.detach().cpu()),
        "shape_invariance_mse": float(invariance_loss.detach().cpu()),
    }
    return loss, metrics


def random_augmentation_transform(batch_size: int, config: CanonicalVoxelAEConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    shift = (torch.rand(batch_size, 3, device=device, dtype=dtype) * 2.0 - 1.0) * 0.12
    theta_xy = (torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.0 - 1.0) * math.pi
    theta_time = (torch.rand(batch_size, 1, device=device, dtype=dtype) * 2.0 - 1.0) * 0.30
    scale = 0.85 + 0.30 * torch.rand(batch_size, 1, device=device, dtype=dtype)
    energy = 0.85 + 0.30 * torch.rand(batch_size, 1, device=device, dtype=dtype)
    return torch.cat([shift, theta_xy, theta_time, scale, energy], dim=1)


def evaluate_canonical_model(
    model: CanonicalTransformVoxelAE,
    dataset: VoxelSparseCache,
    *,
    model_config: CanonicalVoxelAEConfig,
    train_config: CanonicalTrainConfig,
    batch_size: int,
    max_batches: int,
    device: str | torch.device,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {}
    sampler = BucketBalancedVoxelSampler(dataset, seed=train_config.seed + 1009)
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_batches):
            target = sampler.batch(batch_size, device)
            moments = volume_moments(target, model_config)
            canonical_target = transform_density(_normalize_density(target), moments["target_transform"], inverse=False, renormalize=True)
            output = model(target)
            _, metrics = canonical_loss(
                output,
                target,
                model_config,
                train_config,
                target_transform=moments["target_transform"],
                canonical_target=canonical_target,
            )
            rows.append(metrics)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _set_torch_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_compatible_state_dict(model: CanonicalTransformVoxelAE, checkpoint_path: Path, device: str | torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain model_state_dict")
    current = model.state_dict()
    mismatches = [key for key, value in state.items() if key not in current or tuple(current[key].shape) != tuple(value.shape)]
    if mismatches:
        preview = ", ".join(mismatches[:5])
        raise ValueError(f"Checkpoint {checkpoint_path} is incompatible with this model; mismatched tensors: {preview}")
    model.load_state_dict(state)
    return checkpoint


def scan_energy_l2_losses(
    model: CanonicalTransformVoxelAE,
    dataset: VoxelSparseCache,
    stage: VoxelAEStage,
    hardmine_config: HardMineL2Config,
    *,
    batch_size: int,
    device: str | torch.device,
    seed: int,
    mine_clean_loss: bool,
    row_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray | float]:
    indices = np.asarray(list(row_indices) if row_indices is not None else np.arange(len(dataset.rows)), dtype=np.int64)
    corrupted_losses = np.empty(indices.shape[0], dtype=np.float32)
    clean_losses = np.empty(indices.shape[0], dtype=np.float32) if mine_clean_loss else None
    model.eval()
    start_time = time.perf_counter()
    device_type = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for batch_start in range(0, indices.shape[0], batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            rows = [dataset.rows[int(index)] for index in batch_indices]
            target = dataset.rows_to_dense(rows, device=device)
            _set_torch_random_seed(seed + int(batch_start))
            corrupted = corrupt_voxel_batch(
                target,
                blur_kernel=stage.blur_kernel,
                blur_mix=stage.blur_mix,
                noise_std=stage.noise_std,
                voxel_dropout=stage.voxel_dropout,
            )
            with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                output = model(corrupted)
                if hardmine_config.loss_mode == "relative-l2":
                    target_4d = ensure_voxel_channel(target)[:, 0]
                    reconstruction = ensure_voxel_channel(output["reconstruction"])[:, 0]
                    diff = reconstruction - target_4d
                    corrupted_per_particle = torch.sqrt(torch.sum(diff * diff, dim=(1, 2, 3))) / torch.sqrt(
                        torch.sum(target_4d * target_4d, dim=(1, 2, 3))
                    ).clamp_min(1e-8)
                else:
                    corrupted_per_particle = per_particle_energy_weighted_l2(
                        output["reconstruction"],
                        target,
                        background_weight=hardmine_config.background_weight,
                        support_weight=hardmine_config.support_weight,
                        energy_weight=hardmine_config.energy_weight,
                    )
            corrupted_losses[batch_start : batch_start + batch_indices.shape[0]] = corrupted_per_particle.detach().float().cpu().numpy()
            if clean_losses is not None:
                with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                    clean_output = model(target)
                    if hardmine_config.loss_mode == "relative-l2":
                        target_4d = ensure_voxel_channel(target)[:, 0]
                        reconstruction = ensure_voxel_channel(clean_output["reconstruction"])[:, 0]
                        diff = reconstruction - target_4d
                        clean_per_particle = torch.sqrt(torch.sum(diff * diff, dim=(1, 2, 3))) / torch.sqrt(
                            torch.sum(target_4d * target_4d, dim=(1, 2, 3))
                        ).clamp_min(1e-8)
                    else:
                        clean_per_particle = per_particle_energy_weighted_l2(
                            clean_output["reconstruction"],
                            target,
                            background_weight=hardmine_config.background_weight,
                            support_weight=hardmine_config.support_weight,
                            energy_weight=hardmine_config.energy_weight,
                        )
                clean_losses[batch_start : batch_start + batch_indices.shape[0]] = clean_per_particle.detach().float().cpu().numpy()
    result: dict[str, np.ndarray | float] = {
        "row_indices": indices,
        "corrupted_l2": corrupted_losses,
        "scan_runtime_s": time.perf_counter() - start_time,
    }
    if clean_losses is not None:
        result["clean_l2"] = clean_losses
    return result


def write_hard_mine_selection(
    path: Path,
    dataset: VoxelSparseCache,
    selected_indices: np.ndarray,
    corrupted_losses: np.ndarray,
    clean_losses: np.ndarray | None,
) -> None:
    rows: list[dict[str, object]] = []
    clean_lookup = clean_losses if clean_losses is not None else np.full_like(corrupted_losses, np.nan)
    for rank, row_index in enumerate(selected_indices.tolist(), start=1):
        source_row = dataset.rows[int(row_index)]
        rows.append(
            {
                "rank": rank,
                "row_index": int(row_index),
                "corrupted_l2": float(corrupted_losses[int(row_index)]),
                "clean_l2": float(clean_lookup[int(row_index)]),
                "hit_bucket": hit_count_bucket(source_row),
                "n_hits": source_row.get("n_hits", ""),
                "source_path": source_row.get("source_path", ""),
                "source_npz": source_row.get("source_npz", ""),
                "particle_id": source_row.get("particle_id", ""),
                "particle_index": source_row.get("particle_index", ""),
            }
        )
    write_dict_rows(path, rows)


def evaluate_energy_l2_model(
    model: CanonicalTransformVoxelAE,
    dataset: VoxelSparseCache,
    stage: VoxelAEStage,
    hardmine_config: HardMineL2Config,
    *,
    batch_size: int,
    max_batches: int,
    device: str | torch.device,
    corrupted: bool,
    seed: int,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {}
    sampler = BucketBalancedVoxelSampler(dataset, seed=seed)
    rows: list[dict[str, float]] = []
    model.eval()
    device_type = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for batch_idx in range(max_batches):
            target = sampler.batch(batch_size, device)
            if corrupted:
                _set_torch_random_seed(seed + batch_idx)
                model_input = corrupt_voxel_batch(
                    target,
                    blur_kernel=stage.blur_kernel,
                    blur_mix=stage.blur_mix,
                    noise_std=stage.noise_std,
                    voxel_dropout=stage.voxel_dropout,
                )
            else:
                model_input = target
            with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                output = model(model_input)
                _, metrics = hardmine_reconstruction_loss(output, target, hardmine_config)
            rows.append(metrics)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _train_energy_l2_steps(
    model: CanonicalTransformVoxelAE,
    sampler: RowSubsetVoxelSampler,
    val_data: VoxelSparseCache,
    *,
    stage: VoxelAEStage,
    hardmine_config: HardMineL2Config,
    train_config: CanonicalTrainConfig,
    device: str | torch.device,
    start_step: int,
    phase_index: int,
    metrics_rows: list[dict[str, object]],
    metrics_path: Path,
    verbose: bool,
) -> tuple[int, dict[str, object], dict[str, torch.Tensor] | None]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    current_lr = train_config.learning_rate
    global_step = start_step
    stage_best_metric = float("inf")
    stage_best_step = global_step
    best_state = None
    lr_no_improve = 0
    evals = 0
    stop_reason = "max_steps"
    stage_start = time.perf_counter()
    device_type = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"

    for stage_step in range(1, train_config.max_steps_per_stage + 1):
        global_step += 1
        target = sampler.batch(train_config.batch_size, device)
        model_input = corrupt_voxel_batch(
            target,
            blur_kernel=stage.blur_kernel,
            blur_mix=stage.blur_mix,
            noise_std=stage.noise_std,
            voxel_dropout=stage.voxel_dropout,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type, enabled=(train_config.amp and device_type == "cuda")):
            output = model(model_input)
            loss, metrics = hardmine_reconstruction_loss(output, target, hardmine_config)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if global_step == 1 or stage_step % train_config.eval_interval == 0:
            val_metrics = evaluate_energy_l2_model(
                model,
                val_data,
                stage,
                hardmine_config,
                batch_size=train_config.batch_size,
                max_batches=train_config.max_eval_batches,
                device=device,
                corrupted=True,
                seed=train_config.seed + 20_000 + phase_index * 101,
            )
            clean_val_metrics = evaluate_energy_l2_model(
                model,
                val_data,
                stage,
                hardmine_config,
                batch_size=train_config.batch_size,
                max_batches=max(1, min(4, train_config.max_eval_batches)),
                device=device,
                corrupted=False,
                seed=train_config.seed + 30_000 + phase_index * 101,
            )
            monitor = float(val_metrics.get("loss", metrics["loss"]))
            evals += 1
            row = {
                "step": global_step,
                "phase": phase_index,
                "phase_step": stage_step,
                "lr": current_lr,
                "blur_kernel": stage.blur_kernel,
                "blur_mix": stage.blur_mix,
                "noise_std": stage.noise_std,
                "voxel_dropout": stage.voxel_dropout,
                **metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"clean_val_{key}": value for key, value in clean_val_metrics.items()},
            }
            metrics_rows.append(row)
            write_dict_rows(metrics_path, metrics_rows)
            if verbose:
                print(
                    f"step {global_step} [hardmine phase {phase_index}]: loss={metrics['loss']:.6g}, "
                    f"val={monitor:.6g}, clean_val={clean_val_metrics.get('loss', float('nan')):.6g}, lr={current_lr:.2e}",
                    flush=True,
                )
            improved = monitor < stage_best_metric - train_config.early_stop_min_delta
            if improved:
                stage_best_metric = monitor
                stage_best_step = global_step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                lr_no_improve = 0
            else:
                lr_no_improve += 1
            if lr_no_improve >= train_config.lr_patience_evals:
                current_lr *= train_config.lr_decay
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                lr_no_improve = 0
                if current_lr < train_config.stop_lr_below and stage_step >= train_config.min_steps_per_stage:
                    stop_reason = f"lr_below_{train_config.stop_lr_below:g}"
                    break
    phase_summary = {
        "phase": phase_index,
        "start_step": start_step + 1,
        "end_step": global_step,
        "actual_steps": global_step - start_step,
        "evals": evals,
        "best_step": stage_best_step,
        "best_val_loss": stage_best_metric,
        "stop_reason": stop_reason,
        "initial_lr": train_config.learning_rate,
        "final_lr": current_lr,
        "blur_kernel": stage.blur_kernel,
        "blur_mix": stage.blur_mix,
        "noise_std": stage.noise_std,
        "voxel_dropout": stage.voxel_dropout,
        "duration_s": time.perf_counter() - stage_start,
    }
    return global_step, phase_summary, best_state


def _clone_model(model: CanonicalTransformVoxelAE, device: str | torch.device) -> CanonicalTransformVoxelAE:
    cloned = CanonicalTransformVoxelAE(model.config).to(device)
    cloned.load_state_dict({key: value.detach().clone() for key, value in model.state_dict().items()})
    return cloned


def train_hard_mined_l2_canonical_voxel_ae_run(
    cache_dir: str | Path,
    run_dir: str | Path,
    *,
    model_config: CanonicalVoxelAEConfig,
    grid_config: VoxelGridConfig,
    train_config: CanonicalTrainConfig,
    hardmine_config: HardMineL2Config,
    device: str = "cuda",
    init_checkpoint: str | Path | None = None,
    max_train_items: int | None = None,
    max_val_items: int | None = None,
    max_test_items: int | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    mining_dir = run / "mining"
    mining_dir.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    _set_torch_random_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)

    train_data = VoxelSparseCache(cache_dir, split="train", grid_config=grid_config, max_items=max_train_items, seed=train_config.seed)
    val_data = VoxelSparseCache(cache_dir, split="val", grid_config=grid_config, max_items=max_val_items, seed=train_config.seed)
    test_data = VoxelSparseCache(cache_dir, split="test", grid_config=grid_config, max_items=max_test_items, seed=train_config.seed)
    stage = hardmine_config.fixed_stage

    scratch_model = CanonicalTransformVoxelAE(model_config).to(device)
    warm_model = None
    checkpoint_info: dict[str, object] | None = None
    if init_checkpoint is not None and Path(init_checkpoint).exists():
        warm_model = CanonicalTransformVoxelAE(model_config).to(device)
        checkpoint_info = _load_compatible_state_dict(warm_model, Path(init_checkpoint), device)

    selected_model = warm_model if warm_model is not None else scratch_model
    smoke_summary: dict[str, object] = {"enabled": False, "selected": "warm_start" if warm_model is not None else "scratch"}
    if hardmine_config.smoke_compare_init and warm_model is not None:
        smoke_summary["enabled"] = True
        smoke_scan_indices = np.arange(min(len(train_data.rows), hardmine_config.smoke_scan_items), dtype=np.int64)
        smoke_scan = scan_energy_l2_losses(
            warm_model,
            train_data,
            stage,
            hardmine_config,
            batch_size=hardmine_config.scan_batch_size,
            device=device,
            seed=train_config.seed + 7_000,
            mine_clean_loss=False,
            row_indices=smoke_scan_indices,
        )
        smoke_selected_local = select_top_loss_indices(np.asarray(smoke_scan["corrupted_l2"]), hardmine_config.fraction)
        smoke_selected = np.asarray(smoke_scan["row_indices"])[smoke_selected_local]
        candidates = {
            "warm_start": _clone_model(warm_model, device),
            "scratch": _clone_model(scratch_model, device),
        }
        smoke_results: dict[str, dict[str, float]] = {}
        for name, candidate in candidates.items():
            smoke_rows: list[dict[str, object]] = []
            sampler = RowSubsetVoxelSampler(train_data, smoke_selected, seed=train_config.seed + 8_000)
            smoke_train_config = CanonicalTrainConfig(
                seed=train_config.seed,
                batch_size=train_config.batch_size,
                learning_rate=train_config.learning_rate,
                lr_decay=train_config.lr_decay,
                lr_patience_evals=train_config.lr_patience_evals,
                stop_lr_below=train_config.stop_lr_below,
                min_steps_per_stage=max(1, min(train_config.min_steps_per_stage, hardmine_config.smoke_steps)),
                max_steps_per_stage=max(1, hardmine_config.smoke_steps),
                eval_interval=max(1, hardmine_config.smoke_steps),
                max_eval_batches=hardmine_config.smoke_eval_batches,
                weight_decay=train_config.weight_decay,
                grad_clip=train_config.grad_clip,
                amp=train_config.amp,
                early_stop_min_delta=train_config.early_stop_min_delta,
                wall_time_limit_s=train_config.wall_time_limit_s,
                stages=(stage,),
            )
            _, _, _ = _train_energy_l2_steps(
                candidate,
                sampler,
                val_data,
                stage=stage,
                hardmine_config=hardmine_config,
                train_config=smoke_train_config,
                device=device,
                start_step=0,
                phase_index=0,
                metrics_rows=smoke_rows,
                metrics_path=run / f"smoke_{name}_metrics.csv",
                verbose=False,
            )
            val_metrics = evaluate_energy_l2_model(
                candidate,
                val_data,
                stage,
                hardmine_config,
                batch_size=train_config.batch_size,
                max_batches=hardmine_config.smoke_eval_batches,
                device=device,
                corrupted=True,
                seed=train_config.seed + 9_000,
            )
            smoke_results[name] = val_metrics
        warm_loss = float(smoke_results["warm_start"]["loss"])
        scratch_loss = float(smoke_results["scratch"]["loss"])
        if warm_loss <= scratch_loss * (1.0 + hardmine_config.warm_start_prefer_margin):
            selected_model = warm_model
            selected_name = "warm_start"
        else:
            selected_model = scratch_model
            selected_name = "scratch"
        smoke_summary.update({"selected": selected_name, "results": smoke_results, "selected_rows": int(smoke_selected.shape[0])})
    model = selected_model
    (run / "smoke_compare.json").write_text(json.dumps(smoke_summary, indent=2, sort_keys=True), encoding="utf-8")

    metrics_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    mining_rows: list[dict[str, object]] = []
    best_state = None
    best_metric = float("inf")
    best_step = 0
    global_step = 0
    start_time = time.perf_counter()

    for phase_index in range(1, hardmine_config.phases + 1):
        if time.perf_counter() - start_time > train_config.wall_time_limit_s:
            break
        if verbose:
            print(f"hard-mine phase {phase_index}: scanning {len(train_data.rows)} train particles", flush=True)
        scan = scan_energy_l2_losses(
            model,
            train_data,
            stage,
            hardmine_config,
            batch_size=hardmine_config.scan_batch_size,
            device=device,
            seed=train_config.seed + phase_index * 10_000,
            mine_clean_loss=hardmine_config.mine_clean_loss,
        )
        corrupted_l2 = np.asarray(scan["corrupted_l2"], dtype=np.float32)
        clean_l2 = np.asarray(scan["clean_l2"], dtype=np.float32) if "clean_l2" in scan else None
        selected_indices = select_top_loss_indices(corrupted_l2, hardmine_config.fraction)
        write_hard_mine_selection(mining_dir / f"phase_{phase_index:02d}_selected.csv", train_data, selected_indices, corrupted_l2, clean_l2)
        mining_row = {
            "phase": phase_index,
            "rows_scanned": int(corrupted_l2.shape[0]),
            "selected_rows": int(selected_indices.shape[0]),
            "fraction": hardmine_config.fraction,
            "scan_runtime_s": float(scan["scan_runtime_s"]),
            "corrupted_l2_mean": float(np.mean(corrupted_l2)),
            "corrupted_l2_p50": float(np.quantile(corrupted_l2, 0.50)),
            "corrupted_l2_p90": float(np.quantile(corrupted_l2, 0.90)),
            "corrupted_l2_p99": float(np.quantile(corrupted_l2, 0.99)),
            "selected_corrupted_l2_min": float(np.min(corrupted_l2[selected_indices])),
            "selected_corrupted_l2_mean": float(np.mean(corrupted_l2[selected_indices])),
        }
        if clean_l2 is not None:
            mining_row.update(
                {
                    "clean_l2_mean": float(np.mean(clean_l2)),
                    "clean_l2_p90": float(np.quantile(clean_l2, 0.90)),
                    "selected_clean_l2_mean": float(np.mean(clean_l2[selected_indices])),
                }
            )
        mining_rows.append(mining_row)
        write_dict_rows(run / "mining_summary.csv", mining_rows)
        sampler = RowSubsetVoxelSampler(train_data, selected_indices, seed=train_config.seed + phase_index)
        global_step, phase_summary, phase_best_state = _train_energy_l2_steps(
            model,
            sampler,
            val_data,
            stage=stage,
            hardmine_config=hardmine_config,
            train_config=train_config,
            device=device,
            start_step=global_step,
            phase_index=phase_index,
            metrics_rows=metrics_rows,
            metrics_path=run / "metrics.csv",
            verbose=verbose,
        )
        phase_rows.append(phase_summary)
        write_dict_rows(run / "phase_summary.csv", phase_rows)
        if phase_best_state is not None and float(phase_summary["best_val_loss"]) < best_metric:
            best_metric = float(phase_summary["best_val_loss"])
            best_step = int(phase_summary["best_step"])
            best_state = phase_best_state
        if phase_summary["stop_reason"] == "wall_time_limit":
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_corrupted = evaluate_energy_l2_model(
        model,
        test_data,
        stage,
        hardmine_config,
        batch_size=train_config.batch_size,
        max_batches=train_config.max_eval_batches,
        device=device,
        corrupted=True,
        seed=train_config.seed + 40_000,
    )
    test_clean = evaluate_energy_l2_model(
        model,
        test_data,
        stage,
        hardmine_config,
        batch_size=train_config.batch_size,
        max_batches=train_config.max_eval_batches,
        device=device,
        corrupted=False,
        seed=train_config.seed + 50_000,
    )
    checkpoint = run / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "grid_config": asdict(grid_config),
            "train_config": {
                **{key: value for key, value in asdict(train_config).items() if key != "stages"},
                "stages": [asdict(stage)],
            },
            "hardmine_config": asdict(hardmine_config),
            "transform_names": CANONICAL_TRANSFORM_NAMES,
            "best_step": best_step,
            "best_val_loss": best_metric,
            "test_metrics": {"corrupted": test_corrupted, "clean": test_clean},
            "model_type": "canonical_transform_voxel_ae_hardmine_l2",
            "loss_mode": hardmine_config.loss_mode,
            "phase_summary": phase_rows,
            "smoke_compare": smoke_summary,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
            "init_checkpoint_info": {
                "model_type": checkpoint_info.get("model_type") if checkpoint_info else None,
                "best_step": checkpoint_info.get("best_step") if checkpoint_info else None,
                "best_val_loss": checkpoint_info.get("best_val_loss") if checkpoint_info else None,
            },
        },
        checkpoint,
    )
    summary = {
        "run": run.name,
        "checkpoint": checkpoint.as_posix(),
        "metrics": (run / "metrics.csv").as_posix(),
        "phase_summary": (run / "phase_summary.csv").as_posix(),
        "mining_summary": (run / "mining_summary.csv").as_posix(),
        "smoke_compare": (run / "smoke_compare.json").as_posix(),
        "best_step": best_step,
        "best_val_loss": best_metric,
        "duration_s": time.perf_counter() - start_time,
        "model_type": "canonical_transform_voxel_ae_hardmine_l2",
        "loss_mode": hardmine_config.loss_mode,
        "transform_names": list(CANONICAL_TRANSFORM_NAMES),
        "model_config": asdict(model_config),
        "grid_config": asdict(grid_config),
        "train_config": {
            **{key: value for key, value in asdict(train_config).items() if key != "stages"},
            "stages": [asdict(stage)],
        },
        "hardmine_config": asdict(hardmine_config),
        "smoke_summary": smoke_summary,
        **{f"test_corrupted_{key}": value for key, value in test_corrupted.items()},
        **{f"test_clean_{key}": value for key, value in test_clean.items()},
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_hardmine_l2_training_report(run_dir: str | Path, out: str | Path, asset_dir: str | Path) -> Path:
    run = Path(run_dir)
    out_path = Path(out)
    asset_path = Path(asset_dir)
    asset_path.mkdir(parents=True, exist_ok=True)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    metrics = _read_csv_dicts(run / "metrics.csv")
    phase_rows = _read_csv_dicts(run / "phase_summary.csv")
    mining_rows = _read_csv_dicts(run / "mining_summary.csv")
    plot_rel = ""
    if metrics:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [int(float(row["step"])) for row in metrics]
        train_loss = [float(row["loss"]) for row in metrics]
        val_loss = [float(row.get("val_loss", row.get("val_energy_weighted_l2", "nan"))) for row in metrics]
        clean_val = [float(row.get("clean_val_loss", row.get("clean_val_energy_weighted_l2", "nan"))) for row in metrics]
        lr = [float(row["lr"]) for row in metrics]
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), constrained_layout=True, sharex=True)
        axes[0].plot(steps, train_loss, label="train corrupted L2", linewidth=1.3)
        axes[0].plot(steps, val_loss, label="val corrupted L2", linewidth=1.3)
        axes[0].plot(steps, clean_val, label="val clean L2", linewidth=1.0)
        axes[0].set_ylabel("energy-weighted L2")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(steps, lr, color="tab:orange", linewidth=1.3)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("learning rate")
        axes[1].grid(alpha=0.25)
        plot_path = asset_path / "hardmine_l2_convergence.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_rel = plot_path.relative_to(out_path.parent).as_posix()
    lines = [
        "# Phase 2 Canonical Hard-Mined L2 Training",
        "",
        f"Run directory: `{run.as_posix()}`",
        f"Checkpoint: `{summary.get('checkpoint', '')}`",
        "",
        "This run uses fixed highest corruption and trains only on the highest-loss 10% mined from the train split before each phase. The gradient loss is energy-weighted final reconstruction L2 only.",
        "",
    ]
    if plot_rel:
        lines.extend([f"![hardmine convergence]({plot_rel})", ""])
    lines.extend(
        [
            "## Final Metrics",
            "",
            "| split/input | energy-weighted L2 | p95 | energy rel L1 | mass overlap | support leakage |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| test corrupted | {float(summary.get('test_corrupted_loss', float('nan'))):.6f} | "
                f"{float(summary.get('test_corrupted_energy_weighted_l2_p95', float('nan'))):.6f} | "
                f"{float(summary.get('test_corrupted_energy_relative_l1', float('nan'))):.6f} | "
                f"{float(summary.get('test_corrupted_mass_overlap', float('nan'))):.6f} | "
                f"{float(summary.get('test_corrupted_support_leakage', float('nan'))):.6f} |"
            ),
            (
                f"| test clean | {float(summary.get('test_clean_loss', float('nan'))):.6f} | "
                f"{float(summary.get('test_clean_energy_weighted_l2_p95', float('nan'))):.6f} | "
                f"{float(summary.get('test_clean_energy_relative_l1', float('nan'))):.6f} | "
                f"{float(summary.get('test_clean_mass_overlap', float('nan'))):.6f} | "
                f"{float(summary.get('test_clean_support_leakage', float('nan'))):.6f} |"
            ),
            "",
            "## Mining Phases",
            "",
            "| phase | scanned | selected | selected min L2 | selected mean L2 | scan s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mining_rows:
        lines.append(
            f"| {row.get('phase','')} | {row.get('rows_scanned','')} | {row.get('selected_rows','')} | "
            f"{float(row.get('selected_corrupted_l2_min', 'nan')):.6f} | "
            f"{float(row.get('selected_corrupted_l2_mean', 'nan')):.6f} | "
            f"{float(row.get('scan_runtime_s', 'nan')):.1f} |"
        )
    lines.extend(["", "## Training Phases", "", "| phase | steps | best step | best val L2 | final LR | stop reason | duration s |", "|---:|---:|---:|---:|---:|---|---:|"])
    for row in phase_rows:
        lines.append(
            f"| {row.get('phase','')} | {row.get('actual_steps','')} | {row.get('best_step','')} | "
            f"{float(row.get('best_val_loss', 'nan')):.6f} | {float(row.get('final_lr', 'nan')):.2e} | "
            f"{row.get('stop_reason','')} | {float(row.get('duration_s', 'nan')):.1f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def train_canonical_voxel_ae_run(
    cache_dir: str | Path,
    run_dir: str | Path,
    *,
    model_config: CanonicalVoxelAEConfig,
    grid_config: VoxelGridConfig,
    train_config: CanonicalTrainConfig,
    device: str = "cuda",
    max_train_items: int | None = None,
    max_val_items: int | None = None,
    max_test_items: int | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)

    train_data = VoxelSparseCache(cache_dir, split="train", grid_config=grid_config, max_items=max_train_items, seed=train_config.seed)
    val_data = VoxelSparseCache(cache_dir, split="val", grid_config=grid_config, max_items=max_val_items, seed=train_config.seed)
    test_data = VoxelSparseCache(cache_dir, split="test", grid_config=grid_config, max_items=max_test_items, seed=train_config.seed)
    train_sampler = BucketBalancedVoxelSampler(train_data, seed=train_config.seed)
    model = CanonicalTransformVoxelAE(model_config).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    best_state = None
    best_metric = float("inf")
    best_step = 0
    global_step = 0
    start_time = time.perf_counter()

    for stage_idx, stage in enumerate(train_config.stages, start=1):
        if time.perf_counter() - start_time > train_config.wall_time_limit_s:
            break
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
        current_lr = train_config.learning_rate
        stage_start = time.perf_counter()
        stage_start_step = global_step + 1
        stage_best_metric = float("inf")
        stage_best_step = global_step
        no_improve = 0
        lr_no_improve = 0
        evals = 0
        stop_reason = "max_steps"
        for stage_step in range(1, train_config.max_steps_per_stage + 1):
            global_step += 1
            target = train_sampler.batch(train_config.batch_size, device)
            moments = volume_moments(target, model_config)
            target_transform = moments["target_transform"]
            target_density = _normalize_density(target)
            canonical_target = transform_density(target_density, target_transform, inverse=False, renormalize=True)
            model_input = corrupt_voxel_batch(
                target,
                blur_kernel=stage.blur_kernel,
                blur_mix=stage.blur_mix,
                noise_std=stage.noise_std,
                voxel_dropout=stage.voxel_dropout,
            )
            aug_transform = random_augmentation_transform(target.shape[0], model_config, target.device, target.dtype)
            augmented_clean = transform_energy_volume(target, aug_transform, inverse=True)
            augmented_moments = volume_moments(augmented_clean, model_config)
            augmented_canonical_target = transform_density(
                _normalize_density(augmented_clean),
                augmented_moments["target_transform"],
                inverse=False,
                renormalize=True,
            )
            augmented_input = corrupt_voxel_batch(
                augmented_clean,
                blur_kernel=stage.blur_kernel,
                blur_mix=stage.blur_mix,
                noise_std=stage.noise_std,
                voxel_dropout=stage.voxel_dropout,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available())):
                output = model(model_input)
                augmented_output = model(augmented_input)
                loss, metrics = canonical_loss(
                    output,
                    target,
                    model_config,
                    train_config,
                    augmented_output=augmented_output,
                    target_transform=target_transform,
                    canonical_target=canonical_target,
                )
                augmented_loss, augmented_metrics = canonical_loss(
                    augmented_output,
                    augmented_clean,
                    model_config,
                    train_config,
                    target_transform=augmented_moments["target_transform"],
                    canonical_target=augmented_canonical_target,
                )
                loss = loss + train_config.augmentation_weight * augmented_loss
                metrics = {
                    **metrics,
                    "loss": float(loss.detach().cpu()),
                    **{f"aug_{key}": value for key, value in augmented_metrics.items()},
                }
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if global_step == 1 or stage_step % train_config.eval_interval == 0:
                val_metrics = evaluate_canonical_model(
                    model,
                    val_data,
                    model_config=model_config,
                    train_config=train_config,
                    batch_size=train_config.batch_size,
                    max_batches=train_config.max_eval_batches,
                    device=device,
                )
                monitor = float(val_metrics.get("loss", metrics["loss"]))
                evals += 1
                row = {
                    "step": global_step,
                    "stage": stage.name,
                    "stage_index": stage_idx,
                    "stage_step": stage_step,
                    "lr": current_lr,
                    "blur_kernel": stage.blur_kernel,
                    "blur_mix": stage.blur_mix,
                    "noise_std": stage.noise_std,
                    "voxel_dropout": stage.voxel_dropout,
                    **metrics,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
                rows.append(row)
                write_dict_rows(run / "metrics.csv", rows)
                if verbose:
                    print(
                        f"step {global_step} [{stage.name}]: loss={metrics['loss']:.6g}, "
                        f"val={monitor:.6g}, energy={metrics['energy_relative_l1']:.4g}, lr={current_lr:.2e}",
                        flush=True,
                    )
                improved_global = monitor < best_metric
                improved_stage = monitor < stage_best_metric - train_config.early_stop_min_delta
                if improved_global:
                    best_metric = monitor
                    best_step = global_step
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                if improved_stage:
                    stage_best_metric = monitor
                    stage_best_step = global_step
                    no_improve = 0
                    lr_no_improve = 0
                else:
                    no_improve += 1
                    lr_no_improve += 1
                if lr_no_improve >= train_config.lr_patience_evals:
                    current_lr *= train_config.lr_decay
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr
                    lr_no_improve = 0
                    if current_lr < train_config.stop_lr_below and stage_step >= train_config.min_steps_per_stage:
                        stop_reason = f"lr_below_{train_config.stop_lr_below:g}"
                        break
            if time.perf_counter() - start_time > train_config.wall_time_limit_s:
                stop_reason = "wall_time_limit"
                break
        stage_rows.append(
            {
                "stage": stage.name,
                "stage_index": stage_idx,
                "start_step": stage_start_step,
                "end_step": global_step,
                "actual_steps": global_step - stage_start_step + 1,
                "evals": evals,
                "best_step": stage_best_step,
                "best_val_loss": stage_best_metric,
                "stop_reason": stop_reason,
                "initial_lr": train_config.learning_rate,
                "final_lr": current_lr,
                "blur_kernel": stage.blur_kernel,
                "blur_mix": stage.blur_mix,
                "noise_std": stage.noise_std,
                "voxel_dropout": stage.voxel_dropout,
                "duration_s": time.perf_counter() - stage_start,
            }
        )
        write_dict_rows(run / "stage_summary.csv", stage_rows)
        if stop_reason == "wall_time_limit":
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_canonical_model(
        model,
        test_data,
        model_config=model_config,
        train_config=train_config,
        batch_size=train_config.batch_size,
        max_batches=train_config.max_eval_batches,
        device=device,
    )
    checkpoint = run / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "grid_config": asdict(grid_config),
            "train_config": {
                **{key: value for key, value in asdict(train_config).items() if key != "stages"},
                "stages": [asdict(stage) for stage in train_config.stages],
            },
            "transform_names": CANONICAL_TRANSFORM_NAMES,
            "best_step": best_step,
            "best_val_loss": best_metric,
            "test_metrics": test_metrics,
            "model_type": "canonical_transform_voxel_ae",
            "stage_summary": stage_rows,
        },
        checkpoint,
    )
    summary = {
        "run": run.name,
        "checkpoint": checkpoint.as_posix(),
        "metrics": (run / "metrics.csv").as_posix(),
        "stage_summary": (run / "stage_summary.csv").as_posix(),
        "best_step": best_step,
        "best_val_loss": best_metric,
        "duration_s": time.perf_counter() - start_time,
        "model_type": "canonical_transform_voxel_ae",
        "transform_names": list(CANONICAL_TRANSFORM_NAMES),
        "model_config": asdict(model_config),
        "grid_config": asdict(grid_config),
        "train_config": {
            **{key: value for key, value in asdict(train_config).items() if key != "stages"},
            "stages": [asdict(stage) for stage in train_config.stages],
        },
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _load_canonical_checkpoint(path: Path, device: str) -> tuple[CanonicalTransformVoxelAE, VoxelGridConfig, dict[str, object]]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = CanonicalVoxelAEConfig(**checkpoint["model_config"])
    grid_config = VoxelGridConfig(**checkpoint["grid_config"])
    model = CanonicalTransformVoxelAE(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, grid_config, checkpoint


def _four_time_slices(volume: np.ndarray) -> np.ndarray:
    return np.stack([chunk.sum(axis=0) for chunk in np.array_split(volume, 4, axis=0)], axis=0)


def _render_rows(rows: list[tuple[str, np.ndarray]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = max(max(float(_four_time_slices(volume).max()) for _, volume in rows), 1e-6)
    fig, axes = plt.subplots(len(rows), 4, figsize=(12.0, 2.55 * len(rows)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    images = []
    for row_idx, (label, volume) in enumerate(rows):
        slices = _four_time_slices(volume)
        for col in range(4):
            images.append(axes[row_idx, col].imshow(slices[col], origin="lower", cmap="magma", vmin=0, vmax=vmax))
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col].set_title(f"time block {col + 1}")
            if col == 0:
                axes[row_idx, col].set_ylabel(label)
    fig.colorbar(images[0], ax=axes[:, :], shrink=0.78, label="summed normalized log energy")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_canonical_voxel_gallery(
    checkpoint: str | Path,
    cache: str | Path,
    out: str | Path,
    asset_dir: str | Path,
    *,
    split: str = "test",
    examples: int = 10,
    min_hits: int = 50,
    seed: int = 20260507,
    device: str = "cuda",
) -> Path:
    model, grid_config, _ = _load_canonical_checkpoint(Path(checkpoint), device)
    dataset = VoxelSparseCache(cache, split=split, grid_config=grid_config, seed=seed)
    candidates = [row for row in dataset.rows if int(float(row.get("n_hits", 0))) >= min_hits]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:examples]
    if not selected:
        raise ValueError("No matching examples")
    device_obj = next(model.parameters()).device
    batch = dataset.rows_to_dense(selected, device=device_obj)
    with torch.no_grad():
        output = model(batch)
    original = batch.detach().cpu().numpy()
    recon = output["reconstruction"].detach().cpu().numpy()
    canonical = output["canonical_density"].detach().cpu().numpy()
    transformed_density = output["transformed_density"].detach().cpu().numpy()
    z_transform = output["z_transform"].detach().cpu().numpy()
    asset_path = Path(asset_dir)
    asset_path.mkdir(parents=True, exist_ok=True)
    out_path = Path(out)
    lines = ["# Phase 2 Canonical Transform Voxel AE Gallery", "", f"Checkpoint: `{Path(checkpoint).as_posix()}`", ""]
    lines.append("| # | source | particle | hits | energy target/recon | transform | image |")
    lines.append("|---:|---|---:|---:|---:|---|---|")
    for idx, row in enumerate(selected, start=1):
        image = asset_path / f"canonical_voxel_example_{idx:02d}.png"
        _render_rows(
            [
                ("original", original[idx - 1]),
                ("canonical density", canonical[idx - 1]),
                ("transformed density", transformed_density[idx - 1]),
                ("reconstruction", recon[idx - 1]),
                ("abs error", np.abs(recon[idx - 1] - original[idx - 1])),
            ],
            image,
            f"{Path(row['source_path']).name} particle {row['particle_id']} hits {row['n_hits']}",
        )
        rel = image.relative_to(out_path.parent).as_posix()
        transform_text = ", ".join(f"{name}={value:.3g}" for name, value in zip(CANONICAL_TRANSFORM_NAMES, z_transform[idx - 1], strict=True))
        lines.append(
            f"| {idx} | `{row['source_path']}` | {row['particle_id']} | {row['n_hits']} | "
            f"{original[idx - 1].sum():.3f}/{recon[idx - 1].sum():.3f} | `{transform_text}` | [png]({rel}) |"
        )
    lines.append("")
    for idx in range(1, len(selected) + 1):
        image = asset_path / f"canonical_voxel_example_{idx:02d}.png"
        lines.extend([f"## Example {idx}", "", f"![example {idx}]({image.relative_to(out_path.parent).as_posix()})", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def train_canonical_voxel_ae_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the canonical-transform Phase 2 voxel autoencoder.")
    parser.add_argument("--cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_32x64x64_curriculum_v001"))
    parser.add_argument("--out", type=Path, default=Path("local_data/experiments/phase2_canonical_voxel_ae_v001"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    parser.add_argument("--max-steps-per-stage", type=int, default=20512)
    parser.add_argument("--min-steps-per-stage", type=int, default=4096)
    parser.add_argument("--eval-interval", type=int, default=512)
    parser.add_argument("--max-eval-batches", type=int, default=12)
    parser.add_argument("--wall-time-limit-s", type=float, default=12 * 60 * 60)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--loss-mode", choices=["composite", "energy-weighted-l2", "relative-l2"], default="composite")
    parser.add_argument("--hard-mine-phases", type=int, default=0)
    parser.add_argument("--hard-mine-fraction", type=float, default=0.10)
    parser.add_argument("--hard-mine-scan-batch-size", type=int, default=512)
    parser.add_argument("--fixed-corruption", choices=["highest"], default=None)
    parser.add_argument("--mine-clean-loss", action="store_true")
    parser.add_argument("--smoke-compare-init", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=256)
    parser.add_argument("--smoke-scan-items", type=int, default=20_000)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--grid-t-bins", type=int, default=32)
    parser.add_argument("--grid-y-bins", type=int, default=64)
    parser.add_argument("--grid-x-bins", type=int, default=64)
    parser.add_argument("--grid-time-bin", type=float, default=0.625)
    parser.add_argument("--grid-xy-bin", type=float, default=1.0)
    parser.add_argument("--skip-report-generation", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    grid_config = VoxelGridConfig(
        t_bins=args.grid_t_bins,
        y_bins=args.grid_y_bins,
        x_bins=args.grid_x_bins,
        time_bin=args.grid_time_bin,
        xy_bin=args.grid_xy_bin,
    )
    model_config = CanonicalVoxelAEConfig(t_bins=args.grid_t_bins, y_bins=args.grid_y_bins, x_bins=args.grid_x_bins)
    train_config = CanonicalTrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        max_eval_batches=args.max_eval_batches,
        min_steps_per_stage=args.min_steps_per_stage,
        max_steps_per_stage=args.max_steps_per_stage,
        wall_time_limit_s=args.wall_time_limit_s,
        amp=not args.no_amp,
    )
    if args.loss_mode in {"energy-weighted-l2", "relative-l2"} or args.hard_mine_phases > 0:
        init_checkpoint = args.init_checkpoint
        default_init = Path("local_data/experiments/phase2_canonical_voxel_ae_v002/runs/canonical_transform_voxel_z8_t7/checkpoint.pt")
        if init_checkpoint is None and args.smoke_compare_init and default_init.exists():
            init_checkpoint = default_init
        hardmine_config = HardMineL2Config(
            phases=args.hard_mine_phases or 10,
            fraction=args.hard_mine_fraction,
            scan_batch_size=args.hard_mine_scan_batch_size,
            smoke_steps=args.smoke_steps,
            smoke_scan_items=args.smoke_scan_items,
            mine_clean_loss=args.mine_clean_loss,
            smoke_compare_init=args.smoke_compare_init,
            loss_mode=args.loss_mode if args.loss_mode != "composite" else "energy-weighted-l2",
        )
        summary = train_hard_mined_l2_canonical_voxel_ae_run(
            args.cache,
            args.out / "runs" / "canonical_transform_voxel_z8_t7_hardmine_l2",
            model_config=model_config,
            grid_config=grid_config,
            train_config=train_config,
            hardmine_config=hardmine_config,
            device=args.device,
            init_checkpoint=init_checkpoint,
            max_train_items=args.max_train_items,
            max_val_items=args.max_val_items,
            max_test_items=args.max_test_items,
        )
        summary_path = args.out / "canonical_voxel_hardmine_l2_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "cache": args.cache.as_posix(),
                    "output": args.out.as_posix(),
                    "init_checkpoint": init_checkpoint.as_posix() if init_checkpoint is not None else None,
                    "model_config": asdict(model_config),
                    "grid_config": asdict(grid_config),
                    "train_config": {
                        **{key: value for key, value in asdict(train_config).items() if key != "stages"},
                        "stages": [asdict(hardmine_config.fixed_stage)],
                    },
                    "hardmine_config": asdict(hardmine_config),
                    "run": summary,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if not args.skip_report_generation:
            run_dir = Path(summary["checkpoint"]).parent
            write_hardmine_l2_training_report(
                run_dir,
                Path("experimental_notes/autoencoders/voxel/canonical/phase2-canonical-hardmine-l2-training.md"),
                Path("experimental_notes/assets/phase2_canonical_hardmine_l2/training_v001"),
            )
            render_canonical_voxel_gallery(
                summary["checkpoint"],
                args.cache,
                Path("experimental_notes/autoencoders/voxel/canonical/phase2-canonical-hardmine-l2-gallery.md"),
                Path("experimental_notes/assets/phase2_canonical_hardmine_l2/gallery_v001"),
                device=args.device,
            )
            render_canonical_latent_pairs_main(
                [
                    "--checkpoint",
                    str(summary["checkpoint"]),
                    "--cache",
                    str(args.cache),
                    "--out",
                    "experimental_notes/autoencoders/voxel/canonical/phase2-canonical-hardmine-l2-pairs.md",
                    "--asset-dir",
                    "experimental_notes/assets/phase2_canonical_hardmine_l2/pairs_v001",
                    "--device",
                    str(args.device),
                ]
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    summary = train_canonical_voxel_ae_run(
        args.cache,
        args.out / "runs" / "canonical_transform_voxel_z8_t7",
        model_config=model_config,
        grid_config=grid_config,
        train_config=train_config,
        device=args.device,
        max_train_items=args.max_train_items,
        max_val_items=args.max_val_items,
        max_test_items=args.max_test_items,
    )
    summary_path = args.out / "canonical_voxel_autoencoder_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "cache": args.cache.as_posix(),
                "output": args.out.as_posix(),
                "model_config": asdict(model_config),
                "grid_config": asdict(grid_config),
                "train_config": {
                    **{key: value for key, value in asdict(train_config).items() if key != "stages"},
                    "stages": [asdict(stage) for stage in train_config.stages],
                },
                "run": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_canonical_voxel_gallery_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a canonical-transform voxel AE reconstruction gallery.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_32x64x64_curriculum_v001"))
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/voxel/canonical/phase2-canonical-voxel-ae-gallery.md"))
    parser.add_argument("--asset-dir", type=Path, default=Path("experimental_notes/assets/phase2_canonical_voxel_ae/gallery_v001"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(render_canonical_voxel_gallery(args.checkpoint, args.cache, args.out, args.asset_dir, split=args.split, examples=args.examples, min_hits=args.min_hits, device=args.device))


def render_canonical_latent_pairs_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render explicit-transform latent pairs for a canonical voxel AE.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_32x64x64_curriculum_v001"))
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/voxel/canonical/phase2-canonical-latent-pair-gallery.md"))
    parser.add_argument("--asset-dir", type=Path, default=Path("experimental_notes/assets/phase2_canonical_voxel_ae/pairs_v001"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=5000)
    parser.add_argument("--min-hits", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    model, grid_config, _ = _load_canonical_checkpoint(args.checkpoint, args.device)
    dataset = VoxelSparseCache(args.cache, split=args.split, grid_config=grid_config)
    rows = [row for row in dataset.rows if int(float(row.get("n_hits", 0))) >= args.min_hits]
    random.Random(20260507).shuffle(rows)
    rows = rows[: args.candidate_count]
    device = next(model.parameters()).device
    z_batches: list[np.ndarray] = []
    t_batches: list[np.ndarray] = []
    for start in range(0, len(rows), 256):
        batch = dataset.rows_to_dense(rows[start : start + 256], device=device)
        with torch.no_grad():
            z_shape, z_transform = model.encode(batch)
        z_batches.append(z_shape.detach().cpu().numpy())
        t_batches.append(z_transform.detach().cpu().numpy())
    z = np.vstack(z_batches)
    transforms = np.vstack(t_batches)
    z_norm = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-6)
    from scipy.spatial import cKDTree

    tree = cKDTree(z_norm)
    distances, indices = tree.query(z_norm, k=min(64, len(z_norm)))
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(z_norm)):
        for d, j in zip(np.atleast_1d(distances[i])[1:], np.atleast_1d(indices[i])[1:], strict=False):
            a, b = sorted((i, int(j)))
            candidates.append((float(d), a, b))
    chosen = []
    used: set[int] = set()
    for d, a, b in sorted(set(candidates), key=lambda item: item[0]):
        if a in used or b in used:
            continue
        if abs(float(transforms[a, 3] - transforms[b, 3])) < 0.3 and abs(float(transforms[a, 4] - transforms[b, 4])) < 0.1:
            continue
        used.update({a, b})
        chosen.append((d, a, b))
        if len(chosen) >= args.pairs:
            break
    selected = sorted({idx for _, a, b in chosen for idx in (a, b)})
    dense = dataset.rows_to_dense([rows[idx] for idx in selected], device=device)
    with torch.no_grad():
        out = model(dense)
    dense_np = {idx: dense[pos].detach().cpu().numpy() for pos, idx in enumerate(selected)}
    recon_np = {idx: out["reconstruction"][pos].detach().cpu().numpy() for pos, idx in enumerate(selected)}
    canonical_np = {idx: out["canonical_density"][pos].detach().cpu().numpy() for pos, idx in enumerate(selected)}
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Phase 2 Canonical Latent Pair Gallery", "", f"Checkpoint: `{args.checkpoint.as_posix()}`", ""]
    lines.append("| pair | shape dist | d theta_xy | d theta_time | hits A/B | image |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    records = []
    for pair_idx, (d, a, b) in enumerate(chosen, start=1):
        image = args.asset_dir / f"canonical_latent_pair_{pair_idx:02d}.png"
        _render_rows(
            [
                ("A original", dense_np[a]),
                ("A canonical", canonical_np[a]),
                ("A recon", recon_np[a]),
                ("B original", dense_np[b]),
                ("B canonical", canonical_np[b]),
                ("B recon", recon_np[b]),
            ],
            image,
            f"shape distance {d:.4g}",
        )
        rel = image.relative_to(args.out.parent).as_posix()
        lines.append(
            f"| {pair_idx} | {d:.4f} | {abs(float(transforms[a,3]-transforms[b,3])):.3f} | "
            f"{abs(float(transforms[a,4]-transforms[b,4])):.3f} | {rows[a]['n_hits']}/{rows[b]['n_hits']} | [png]({rel}) |"
        )
        records.append((pair_idx, a, b, rel))
    for pair_idx, a, b, rel in records:
        lines.extend(["", f"## Pair {pair_idx}", "", f"![pair {pair_idx}]({rel})", ""])
        lines.append(f"- A transform: `{', '.join(f'{name}={value:.3g}' for name, value in zip(CANONICAL_TRANSFORM_NAMES, transforms[a], strict=True))}`")
        lines.append(f"- B transform: `{', '.join(f'{name}={value:.3g}' for name, value in zip(CANONICAL_TRANSFORM_NAMES, transforms[b], strict=True))}`")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)


__all__ = [
    "CANONICAL_TRANSFORM_NAMES",
    "CanonicalVoxelAEConfig",
    "CanonicalTrainConfig",
    "HardMineL2Config",
    "CanonicalTransformVoxelAE",
    "BucketBalancedVoxelSampler",
    "RowSubsetVoxelSampler",
    "affine_grid_from_transform",
    "canonical_loss",
    "energy_weighted_l2_loss",
    "per_particle_energy_weighted_l2",
    "render_canonical_voxel_gallery",
    "scan_energy_l2_losses",
    "select_top_loss_indices",
    "train_hard_mined_l2_canonical_voxel_ae_run",
    "train_canonical_voxel_ae_run",
    "transform_density",
    "transform_energy_volume",
    "volume_moments",
]

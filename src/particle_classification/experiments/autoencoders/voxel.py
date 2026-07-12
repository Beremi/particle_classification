from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...dbscan.pipeline import write_dict_rows
from ..neural_separator.dataset import assign_group_split, discover_particle_shards


VOXEL_AE_SCHEMA_VERSION = "phase2_voxel_energy_v1"


@dataclass(frozen=True)
class VoxelGridConfig:
    """Centered fixed-size voxel grid for one particle."""

    t_bins: int = 32
    y_bins: int = 64
    x_bins: int = 64
    time_bin: float = 0.625
    xy_bin: float = 1.0
    energy_norm: float = math.log1p(1023.0)
    min_particle_hits: int = 2
    seed: int = 20260507
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    chunk_size: int = 4096
    max_particles: int | None = None

    @property
    def flat_dim(self) -> int:
        return self.t_bins * self.y_bins * self.x_bins


@dataclass(frozen=True)
class VoxelMLPConfig:
    """Patch-MLP autoencoder with an explicitly split latent layer."""

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
    aux_latent_dim: int = 7
    dropout: float = 0.03
    architecture: str = "patch_mlp"
    output_activation: str = "relu"
    output_bias_init: float = 0.0

    @property
    def latent_dim(self) -> int:
        return self.shape_latent_dim + self.aux_latent_dim

    @property
    def flat_dim(self) -> int:
        return self.t_bins * self.y_bins * self.x_bins


@dataclass(frozen=True)
class VoxelAEStage:
    name: str
    steps: int
    blur_kernel: int
    noise_std: float
    voxel_dropout: float
    blur_mix: float = 1.0
    learning_rate_scale: float = 1.0


@dataclass(frozen=True)
class VoxelAETrainConfig:
    seed: int = 20260507
    batch_size: int = 64
    learning_rate: float = 4e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 2e-4
    grad_clip: float = 1.0
    eval_interval: int = 250
    max_eval_batches: int = 16
    amp: bool = True
    occupied_weight: float = 0.0
    energy_weight: float = 0.0
    lr_schedule: str = "cosine"
    phase_min_steps: int = 0
    phase_patience_evals: int = 0
    lr_patience_evals: int = 0
    lr_decay: float = 0.5
    keep_lr_across_stages: bool = False
    stop_lr_below: float = 0.0
    early_stop_min_delta: float = 0.0
    stages: tuple[VoxelAEStage, ...] = (
        VoxelAEStage("coarse_blur_denoise", steps=2_000, blur_kernel=5, noise_std=0.04, voxel_dropout=0.08),
        VoxelAEStage("medium_blur_denoise", steps=3_000, blur_kernel=3, noise_std=0.02, voxel_dropout=0.03),
        VoxelAEStage("sharp_finetune", steps=3_000, blur_kernel=1, noise_std=0.004, voxel_dropout=0.0, learning_rate_scale=0.45),
    )

    @property
    def total_steps(self) -> int:
        return sum(stage.steps for stage in self.stages)


def make_decay_voxel_curriculum_stages(
    *,
    phases: int = 10,
    steps_per_phase: int = 1_500,
    start_blur_kernel: int = 5,
    start_noise_std: float = 0.04,
    start_voxel_dropout: float = 0.08,
    start_blur_mix: float = 1.0,
    final_fraction: float = 0.1,
    learning_rate_scale: float = 1.0,
) -> tuple[VoxelAEStage, ...]:
    """Create a blur/noise curriculum that never reaches a perfectly clean input.

    ``blur_kernel`` is discrete, so the curriculum decays the continuous
    ``blur_mix`` by ``final_fraction`` while keeping a small nonzero blur in the
    last phase. Noise and dropout use the same geometric decay.
    """

    if phases < 1:
        raise ValueError("phases must be at least 1")
    if start_blur_kernel < 1 or start_blur_kernel % 2 == 0:
        raise ValueError("start_blur_kernel must be a positive odd integer")
    if not (0.0 < final_fraction <= 1.0):
        raise ValueError("final_fraction must be in (0, 1]")
    stages: list[VoxelAEStage] = []
    for phase in range(phases):
        progress = 0.0 if phases == 1 else phase / (phases - 1)
        factor = final_fraction**progress
        blur_kernel = start_blur_kernel if factor >= 0.45 else max(3, start_blur_kernel - 2)
        if blur_kernel % 2 == 0:
            blur_kernel -= 1
        stages.append(
            VoxelAEStage(
                name=f"phase_{phase + 1:02d}_blur{blur_kernel}_factor{factor:.3f}",
                steps=steps_per_phase,
                blur_kernel=blur_kernel,
                blur_mix=max(float(start_blur_mix * factor), 1e-6),
                noise_std=float(start_noise_std * factor),
                voxel_dropout=float(start_voxel_dropout * factor),
                learning_rate_scale=learning_rate_scale,
            )
        )
    return tuple(stages)


def voxelize_particle_energy(
    x: np.ndarray,
    y: np.ndarray,
    time_ticks: np.ndarray,
    log_energy: np.ndarray,
    config: VoxelGridConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Convert one particle into sparse centered voxel indices and values.

    The dense logical tensor shape is ``[T, Y, X]``. Values are summed
    normalized log-ToT energy. Hits outside the centered crop are ignored.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    time_ticks = np.asarray(time_ticks, dtype=np.float64)
    log_energy = np.asarray(log_energy, dtype=np.float64)
    n_hits = int(x.shape[0])
    if n_hits == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), empty_voxel_metadata()

    center_x = 0.5 * (float(np.min(x)) + float(np.max(x)))
    center_y = 0.5 * (float(np.min(y)) + float(np.max(y)))
    center_t = 0.5 * (float(np.min(time_ticks)) + float(np.max(time_ticks)))

    x_idx = np.floor((x - center_x) / config.xy_bin + config.x_bins / 2.0).astype(np.int64)
    y_idx = np.floor((y - center_y) / config.xy_bin + config.y_bins / 2.0).astype(np.int64)
    t_idx = np.floor((time_ticks - center_t) / config.time_bin + config.t_bins / 2.0).astype(np.int64)
    valid = (
        (x_idx >= 0)
        & (x_idx < config.x_bins)
        & (y_idx >= 0)
        & (y_idx < config.y_bins)
        & (t_idx >= 0)
        & (t_idx < config.t_bins)
    )
    values = np.clip(log_energy, 0.0, None) / max(float(config.energy_norm), 1e-6)
    total_energy = float(np.sum(values))
    kept_energy = float(np.sum(values[valid]))
    kept_hits = int(np.count_nonzero(valid))
    if kept_hits == 0:
        metadata = {
            "n_hits": n_hits,
            "kept_hits": 0,
            "kept_fraction": 0.0,
            "kept_energy_fraction": 0.0,
            "center_x": center_x,
            "center_y": center_y,
            "center_t": center_t,
            "x_span": float(np.max(x) - np.min(x) + 1.0),
            "y_span": float(np.max(y) - np.min(y) + 1.0),
            "time_span": float(np.max(time_ticks) - np.min(time_ticks)),
        }
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), metadata

    flat = (t_idx[valid] * config.y_bins * config.x_bins + y_idx[valid] * config.x_bins + x_idx[valid]).astype(np.int64)
    unique, inverse = np.unique(flat, return_inverse=True)
    summed = np.zeros(unique.shape[0], dtype=np.float32)
    np.add.at(summed, inverse, values[valid].astype(np.float32))
    metadata = {
        "n_hits": n_hits,
        "kept_hits": kept_hits,
        "kept_fraction": float(kept_hits / max(n_hits, 1)),
        "kept_energy_fraction": float(kept_energy / max(total_energy, 1e-12)),
        "center_x": center_x,
        "center_y": center_y,
        "center_t": center_t,
        "x_span": float(np.max(x) - np.min(x) + 1.0),
        "y_span": float(np.max(y) - np.min(y) + 1.0),
        "time_span": float(np.max(time_ticks) - np.min(time_ticks)),
        "occupied_voxels": int(unique.shape[0]),
        "energy_sum": total_energy,
    }
    return unique.astype(np.int32), summed.astype(np.float32), metadata


def empty_voxel_metadata() -> dict[str, float]:
    return {
        "n_hits": 0,
        "kept_hits": 0,
        "kept_fraction": 0.0,
        "kept_energy_fraction": 0.0,
        "center_x": 0.0,
        "center_y": 0.0,
        "center_t": 0.0,
        "x_span": 0.0,
        "y_span": 0.0,
        "time_span": 0.0,
        "occupied_voxels": 0,
        "energy_sum": 0.0,
    }


def build_voxel_energy_cache(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    manifest: str | Path | None = None,
    config: VoxelGridConfig | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    config = config or VoxelGridConfig()
    output = Path(output_dir)
    chunks_dir = output / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    shards = discover_particle_shards(input_path, manifest)
    rows: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    counts = {"particles": 0, "chunks": 0, "cropped_particles": 0, "total_hits": 0, "kept_hits": 0}

    for shard_idx, shard in enumerate(shards, start=1):
        if config.max_particles is not None and counts["particles"] >= config.max_particles:
            break
        if verbose:
            print(f"[{shard_idx}/{len(shards)}] voxelizing {shard.source_path}", flush=True)
        try:
            with np.load(shard.path, allow_pickle=False) as data:
                hit_x = data["hit_x"].astype(np.float64)
                hit_y = data["hit_y"].astype(np.float64)
                hit_time = data["hit_time"].astype(np.float64)
                hit_energy = data["hit_energy"].astype(np.float64)
                offsets = data["particle_offsets"].astype(np.int64)
                particle_id = data["particle_id"].astype(np.int32)
                for particle_index, pid in enumerate(particle_id.tolist()):
                    if config.max_particles is not None and counts["particles"] >= config.max_particles:
                        break
                    start = int(offsets[particle_index])
                    end = int(offsets[particle_index + 1])
                    if end - start < config.min_particle_hits:
                        continue
                    voxel_index, voxel_value, meta = voxelize_particle_energy(
                        hit_x[start:end],
                        hit_y[start:end],
                        hit_time[start:end],
                        hit_energy[start:end],
                        config,
                    )
                    split_group = shard.source_path
                    split = assign_group_split(split_group, seed=config.seed, val_fraction=config.val_fraction, test_fraction=config.test_fraction)
                    pending.append(
                        {
                            "voxel_index": voxel_index,
                            "voxel_value": voxel_value,
                            "source_npz": shard.path.as_posix(),
                            "source_path": shard.source_path,
                            "split": split,
                            "split_group": split_group,
                            "particle_id": int(pid),
                            "particle_index": int(particle_index),
                            **meta,
                        }
                    )
                    counts["particles"] += 1
                    counts["total_hits"] += int(meta["n_hits"])
                    counts["kept_hits"] += int(meta["kept_hits"])
                    if float(meta["kept_fraction"]) < 0.999:
                        counts["cropped_particles"] += 1
                    if len(pending) >= config.chunk_size:
                        rows.extend(flush_voxel_chunk(pending, chunks_dir, counts["chunks"], config))
                        counts["chunks"] += 1
                        pending.clear()
        except Exception as exc:
            rows.append({"status": f"{type(exc).__name__}: {exc}", "split": "error", "source_path": shard.source_path, "source_npz": shard.path.as_posix()})

    if pending:
        rows.extend(flush_voxel_chunk(pending, chunks_dir, counts["chunks"], config))
        counts["chunks"] += 1
        pending.clear()

    manifest_path = output / "manifest.csv"
    write_dict_rows(manifest_path, rows)
    summary = {
        "schema_version": VOXEL_AE_SCHEMA_VERSION,
        "input": Path(input_path).as_posix(),
        "output": output.as_posix(),
        "manifest": manifest_path.as_posix(),
        "grid_config": asdict(config),
        **counts,
        "kept_hit_fraction": float(counts["kept_hits"] / max(counts["total_hits"], 1)),
        "splits": {split: sum(1 for row in rows if row.get("status") == "ok" and row.get("split") == split) for split in ["train", "val", "test"]},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def flush_voxel_chunk(views: list[dict[str, object]], chunks_dir: Path, chunk_id: int, config: VoxelGridConfig) -> list[dict[str, object]]:
    chunk_path = chunks_dir / f"voxel_chunk_{chunk_id:06d}.npz"
    offsets = np.zeros(len(views) + 1, dtype=np.int64)
    index_arrays = [np.asarray(view["voxel_index"], dtype=np.int32) for view in views]
    value_arrays = [np.asarray(view["voxel_value"], dtype=np.float32) for view in views]
    cursor = 0
    for idx, arr in enumerate(index_arrays, start=1):
        cursor += int(arr.shape[0])
        offsets[idx] = cursor
    voxel_index = np.concatenate(index_arrays).astype(np.int32) if index_arrays else np.empty(0, dtype=np.int32)
    voxel_value = np.concatenate(value_arrays).astype(np.float32) if value_arrays else np.empty(0, dtype=np.float32)
    tmp_path = chunk_path.with_name(f"{chunk_path.name}.tmp.npz")
    np.savez_compressed(
        tmp_path,
        voxel_offsets=offsets,
        voxel_index=voxel_index,
        voxel_value=voxel_value,
        particle_id=np.asarray([int(view["particle_id"]) for view in views], dtype=np.int32),
        particle_index=np.asarray([int(view["particle_index"]) for view in views], dtype=np.int32),
        n_hits=np.asarray([int(view["n_hits"]) for view in views], dtype=np.int32),
        kept_hits=np.asarray([int(view["kept_hits"]) for view in views], dtype=np.int32),
        kept_fraction=np.asarray([float(view["kept_fraction"]) for view in views], dtype=np.float32),
        kept_energy_fraction=np.asarray([float(view["kept_energy_fraction"]) for view in views], dtype=np.float32),
        source_path=np.asarray([str(view["source_path"]) for view in views]),
        source_npz=np.asarray([str(view["source_npz"]) for view in views]),
        grid_config_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
    )
    tmp_path.replace(chunk_path)
    rows: list[dict[str, object]] = []
    for chunk_row, view in enumerate(views):
        rows.append(
            {
                "status": "ok",
                "split": view["split"],
                "split_group": view["split_group"],
                "chunk_path": chunk_path.as_posix(),
                "chunk_row": chunk_row,
                "source_npz": view["source_npz"],
                "source_path": view["source_path"],
                "particle_id": view["particle_id"],
                "particle_index": view["particle_index"],
                "n_hits": view["n_hits"],
                "kept_hits": view["kept_hits"],
                "kept_fraction": view["kept_fraction"],
                "kept_energy_fraction": view["kept_energy_fraction"],
                "occupied_voxels": view.get("occupied_voxels", 0),
                "x_span": view["x_span"],
                "y_span": view["y_span"],
                "time_span": view["time_span"],
                "schema_version": VOXEL_AE_SCHEMA_VERSION,
            }
        )
        for optional_key in ("representative_group", "representative_signature", "representative_source_row", "mirror_axis"):
            if optional_key in view:
                rows[-1][optional_key] = view[optional_key]
    return rows


def load_voxel_manifest(manifest_path: str | Path, *, split: str, max_items: int | None = None, seed: int = 20260507) -> list[dict[str, str]]:
    rng = random.Random(f"{seed}:{Path(manifest_path).as_posix()}:{split}:{max_items}")
    rows: list[dict[str, str]] = []
    seen = 0
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok" or row.get("split") != split:
                continue
            if max_items is None:
                rows.append(row)
                continue
            seen += 1
            if len(rows) < max_items:
                rows.append(row)
            else:
                idx = rng.randrange(seen)
                if idx < max_items:
                    rows[idx] = row
    return rows


class VoxelSparseCache:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str,
        grid_config: VoxelGridConfig,
        max_items: int | None = None,
        seed: int = 20260507,
        max_cached_chunks: int = 32,
    ):
        self.cache_dir = Path(cache_dir)
        self.grid_config = grid_config
        self.rows = load_voxel_manifest(self.cache_dir / "manifest.csv", split=split, max_items=max_items, seed=seed)
        self.rng = random.Random(f"{seed}:{split}:{self.cache_dir.as_posix()}")
        self.max_cached_chunks = max_cached_chunks
        self._chunks: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_chunk(self, path: str) -> dict[str, np.ndarray]:
        if path in self._chunks:
            self._chunks.move_to_end(path)
            return self._chunks[path]
        with np.load(path, allow_pickle=False) as data:
            chunk = {key: data[key] for key in data.files}
        self._chunks[path] = chunk
        while len(self._chunks) > self.max_cached_chunks:
            self._chunks.popitem(last=False)
        return chunk

    def random_batch(self, batch_size: int, device: str | torch.device) -> torch.Tensor:
        if not self.rows:
            raise ValueError(f"No rows available in {self.cache_dir}")
        selected = [self.rows[self.rng.randrange(len(self.rows))] for _ in range(batch_size)]
        return self.rows_to_dense(selected, device=device)

    def rows_to_dense(self, selected: Iterable[dict[str, str]], device: str | torch.device) -> torch.Tensor:
        selected = list(selected)
        dense = torch.zeros((len(selected), self.grid_config.flat_dim), dtype=torch.float32, device=device)
        for batch_idx, row in enumerate(selected):
            chunk = self._load_chunk(row["chunk_path"])
            chunk_row = int(row["chunk_row"])
            start = int(chunk["voxel_offsets"][chunk_row])
            end = int(chunk["voxel_offsets"][chunk_row + 1])
            if end <= start:
                continue
            idx = torch.as_tensor(chunk["voxel_index"][start:end].astype(np.int64), dtype=torch.long, device=device)
            val = torch.as_tensor(chunk["voxel_value"][start:end].astype(np.float32), dtype=torch.float32, device=device)
            dense[batch_idx].scatter_add_(0, idx, val)
        return dense.view(len(selected), self.grid_config.t_bins, self.grid_config.y_bins, self.grid_config.x_bins)


class VoxelPatchMLPAutoencoder(nn.Module):
    def __init__(self, config: VoxelMLPConfig):
        super().__init__()
        self.config = config
        if config.architecture not in {"patch_mlp", "flat_mlp"}:
            raise ValueError(f"Unsupported voxel AE architecture: {config.architecture}")
        if config.architecture == "patch_mlp":
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
            self.encoder = nn.Sequential(
                nn.Linear(n_patches * config.patch_embed_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
            )
            self.decoder_global = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, n_patches * config.patch_embed_dim),
                nn.SiLU(),
            )
            self.patch_decoder = nn.Sequential(
                nn.Linear(config.patch_embed_dim, config.patch_hidden_dim),
                nn.LayerNorm(config.patch_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.patch_hidden_dim, patch_dim),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(config.flat_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.flat_dim),
            )
        self.shape_head = nn.Linear(config.hidden_dim, config.shape_latent_dim)
        self.aux_head = nn.Linear(config.hidden_dim, config.aux_latent_dim)
        self.apply(init_voxel_weights)
        self.reset_output_layers()

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
        volume = ensure_voxel_channel(volume)
        if self.config.architecture == "patch_mlp":
            patches = self.patchify(volume)
            encoded_patches = self.patch_encoder(patches)
            hidden = self.encoder(encoded_patches.reshape(volume.shape[0], -1))
        else:
            hidden = self.encoder(volume.reshape(volume.shape[0], -1))
        return self.shape_head(hidden), self.aux_head(hidden)

    def decode(self, z_shape: torch.Tensor, z_aux: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_shape, z_aux], dim=-1)
        if self.config.architecture == "patch_mlp":
            patch_embed = self.decoder_global(z).view(z.shape[0], self.n_patches, self.config.patch_embed_dim)
            patches = self.patch_decoder(patch_embed)
            return self.activate_output(self.unpatchify(patches))
        return self.activate_output(self.decoder(z).view(z.shape[0], 1, self.config.t_bins, self.config.y_bins, self.config.x_bins))

    def activate_output(self, raw: torch.Tensor) -> torch.Tensor:
        if self.config.output_activation == "relu":
            return torch.relu(raw)
        if self.config.output_activation == "softplus":
            return F.softplus(raw)
        raise ValueError(f"Unsupported output activation: {self.config.output_activation}")

    def forward(self, volume: torch.Tensor) -> dict[str, torch.Tensor]:
        input_was_4d = volume.ndim == 4
        z_shape, z_aux = self.encode(volume)
        reconstruction = self.decode(z_shape, z_aux)
        if input_was_4d:
            reconstruction = reconstruction[:, 0]
        return {"z_shape": z_shape, "z_aux": z_aux, "reconstruction": reconstruction}

    def reset_output_layers(self) -> None:
        """Start sparse-volume decoders near empty, without changing hidden layers."""

        if self.config.architecture == "patch_mlp":
            output = self.patch_decoder[-1]
        else:
            output = self.decoder[-1]
        if isinstance(output, nn.Linear):
            nn.init.normal_(output.weight, mean=0.0, std=1e-4)
            if output.bias is not None:
                nn.init.constant_(output.bias, self.config.output_bias_init)


def init_voxel_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def ensure_voxel_channel(volume: torch.Tensor) -> torch.Tensor:
    """Return a ``[B, 1, T, Y, X]`` tensor from a logical 3D voxel batch."""

    if volume.ndim == 4:
        return volume.unsqueeze(1)
    if volume.ndim == 5:
        return volume
    raise ValueError(f"Expected voxel tensor [B,T,Y,X] or [B,1,T,Y,X], got {tuple(volume.shape)}")


def corrupt_voxel_batch(
    volume: torch.Tensor,
    *,
    blur_kernel: int,
    noise_std: float,
    voxel_dropout: float,
    blur_mix: float = 1.0,
) -> torch.Tensor:
    input_was_4d = volume.ndim == 4
    volume_5d = ensure_voxel_channel(volume)
    out = volume_5d
    if blur_kernel > 1 and blur_mix > 0:
        if blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd")
        kernel = torch.ones((1, 1, blur_kernel, blur_kernel, blur_kernel), dtype=volume_5d.dtype, device=volume_5d.device)
        kernel = kernel / kernel.sum()
        blurred = F.conv3d(out, kernel, padding=blur_kernel // 2)
        mix = float(np.clip(blur_mix, 0.0, 1.0))
        out = out * (1.0 - mix) + blurred * mix
    if voxel_dropout > 0:
        active = out > 0
        keep = torch.rand_like(out) > voxel_dropout
        out = torch.where(active & ~keep, torch.zeros_like(out), out)
    if noise_std > 0:
        scale = torch.amax(volume_5d, dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-3)
        support = out > 0
        out = torch.where(support, out + torch.randn_like(out) * noise_std * scale, out)
    out = torch.clamp(out, min=0.0)
    return out[:, 0] if input_was_4d else out


def voxel_ae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    occupied_weight: float = 0.0,
    energy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    diff = reconstruction - target
    mse = torch.mean(diff * diff)
    loss = mse
    occupied = target > 0
    if occupied.any():
        occupied_mse = torch.mean(diff[occupied] * diff[occupied])
        if occupied_weight > 0:
            loss = loss + occupied_weight * occupied_mse
    else:
        occupied_mse = torch.zeros((), dtype=target.dtype, device=target.device)
    background = ~occupied
    background_mse = torch.mean(diff[background] * diff[background]) if background.any() else torch.zeros((), dtype=target.dtype, device=target.device)
    reduce_dims = tuple(range(1, target.ndim))
    target_energy = torch.sum(target, dim=reduce_dims).clamp_min(1e-6)
    recon_energy = torch.sum(reconstruction, dim=reduce_dims)
    energy_sum_relative_l1 = torch.mean(torch.abs(recon_energy - target_energy) / target_energy)
    if energy_weight > 0:
        loss = loss + energy_weight * energy_sum_relative_l1
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "occupied_mse": float(occupied_mse.detach().cpu()),
        "background_mse": float(background_mse.detach().cpu()),
        "energy_sum_relative_l1": float(energy_sum_relative_l1.detach().cpu()),
    }


def evaluate_voxel_ae(
    model: VoxelPatchMLPAutoencoder,
    dataset: VoxelSparseCache,
    *,
    batch_size: int,
    max_batches: int,
    device: str | torch.device,
    occupied_weight: float = 0.0,
    energy_weight: float = 0.0,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {}
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_batches):
            target = dataset.random_batch(batch_size, device)
            output = model(target)
            _, metrics = voxel_ae_loss(output["reconstruction"], target, occupied_weight=occupied_weight, energy_weight=energy_weight)
            rows.append(metrics)
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_voxel_ae_run(
    cache_dir: str | Path,
    run_dir: str | Path,
    *,
    model_config: VoxelMLPConfig,
    grid_config: VoxelGridConfig,
    train_config: VoxelAETrainConfig,
    device: str = "cuda",
    max_train_items: int | None = None,
    max_val_items: int | None = None,
    max_test_items: int | None = None,
    init_checkpoint: str | Path | None = None,
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
    model = VoxelPatchMLPAutoencoder(model_config).to(device)
    init_checkpoint_path = Path(init_checkpoint) if init_checkpoint is not None else None
    if init_checkpoint_path is not None:
        checkpoint_data = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        checkpoint_model_config = checkpoint_data.get("model_config", {})
        checkpoint_grid_config = checkpoint_data.get("grid_config", {})
        if checkpoint_model_config != asdict(model_config):
            raise ValueError("init checkpoint model_config does not match requested model_config")
        if checkpoint_grid_config != asdict(grid_config):
            raise ValueError("init checkpoint grid_config does not match requested grid_config")
        model.load_state_dict(checkpoint_data["model_state_dict"])
        if verbose:
            print(f"loaded init checkpoint: {init_checkpoint_path}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    best_state = None
    best_metric = float("inf")
    best_step = 0
    global_step = 0
    start_time = time.perf_counter()
    total_steps = max(train_config.total_steps, 1)
    if train_config.lr_schedule not in {"cosine", "plateau"}:
        raise ValueError(f"Unsupported lr_schedule: {train_config.lr_schedule}")
    current_lr = max(train_config.min_learning_rate, train_config.learning_rate)
    stop_all = False
    for stage in train_config.stages:
        if stop_all:
            break
        stage_start_time = time.perf_counter()
        stage_start_step = global_step + 1
        stage_best_metric = float("inf")
        stage_best_step = global_step
        stage_evals = 0
        no_improve_evals = 0
        lr_no_improve_evals = 0
        if train_config.lr_schedule == "cosine" or not train_config.keep_lr_across_stages:
            current_lr = max(train_config.min_learning_rate, train_config.learning_rate * stage.learning_rate_scale)
        stop_reason = "max_steps"
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        for stage_step in range(1, stage.steps + 1):
            global_step += 1
            if train_config.lr_schedule == "cosine":
                progress = min(global_step / total_steps, 1.0)
                current_lr = train_config.min_learning_rate + (
                    train_config.learning_rate * stage.learning_rate_scale - train_config.min_learning_rate
                ) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            target = train_data.random_batch(train_config.batch_size, device)
            model_input = corrupt_voxel_batch(
                target,
                blur_kernel=stage.blur_kernel,
                blur_mix=stage.blur_mix,
                noise_std=stage.noise_std,
                voxel_dropout=stage.voxel_dropout,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available())):
                output = model(model_input)
                loss, metrics = voxel_ae_loss(
                    output["reconstruction"],
                    target,
                    occupied_weight=train_config.occupied_weight,
                    energy_weight=train_config.energy_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if global_step == 1 or global_step % train_config.eval_interval == 0 or stage_step == stage.steps:
                val_metrics = evaluate_voxel_ae(
                    model,
                    val_data,
                    batch_size=train_config.batch_size,
                    max_batches=train_config.max_eval_batches,
                    device=device,
                    occupied_weight=train_config.occupied_weight,
                    energy_weight=train_config.energy_weight,
                )
                row = {
                    "step": global_step,
                    "stage": stage.name,
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
                monitor = float(val_metrics.get("loss", metrics["loss"]))
                stage_evals += 1
                if verbose:
                    print(
                        f"step {global_step}/{total_steps} [{stage.name}]: loss={metrics['loss']:.6g}, "
                        f"val_loss={monitor:.6g}, occ={metrics['occupied_mse']:.6g}, lr={current_lr:.2e}",
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
                    no_improve_evals = 0
                    lr_no_improve_evals = 0
                else:
                    no_improve_evals += 1
                    lr_no_improve_evals += 1
                if (
                    train_config.lr_schedule == "plateau"
                    and train_config.lr_patience_evals > 0
                    and lr_no_improve_evals >= train_config.lr_patience_evals
                ):
                    current_lr = current_lr * train_config.lr_decay
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr
                    lr_no_improve_evals = 0
                    if train_config.stop_lr_below > 0 and current_lr < train_config.stop_lr_below:
                        stop_reason = f"lr_below_{train_config.stop_lr_below:g}"
                        stop_all = True
                        if verbose:
                            print(f"stopping: learning rate {current_lr:.2e} fell below {train_config.stop_lr_below:.2e}", flush=True)
                        break
                if (
                    train_config.phase_patience_evals > 0
                    and no_improve_evals >= train_config.phase_patience_evals
                    and stage_step >= max(train_config.phase_min_steps, train_config.eval_interval)
                ):
                    stop_reason = "plateau"
                    if verbose:
                        print(f"stage {stage.name} stopped at step {stage_step}/{stage.steps} after validation plateau", flush=True)
                    break
        stage_rows.append(
            {
                "stage": stage.name,
                "start_step": stage_start_step,
                "end_step": global_step,
                "planned_steps": stage.steps,
                "actual_steps": global_step - stage_start_step + 1,
                "evals": stage_evals,
                "best_step": stage_best_step,
                "best_val_loss": stage_best_metric,
                "stop_reason": stop_reason,
                "blur_kernel": stage.blur_kernel,
                "blur_mix": stage.blur_mix,
                "noise_std": stage.noise_std,
                "voxel_dropout": stage.voxel_dropout,
                "duration_s": time.perf_counter() - stage_start_time,
            }
        )
        write_dict_rows(run / "stage_summary.csv", stage_rows)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_voxel_ae(
        model,
        test_data,
        batch_size=train_config.batch_size,
        max_batches=train_config.max_eval_batches,
        device=device,
        occupied_weight=train_config.occupied_weight,
        energy_weight=train_config.energy_weight,
    )
    checkpoint = run / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "grid_config": asdict(grid_config),
            "init_checkpoint": init_checkpoint_path.as_posix() if init_checkpoint_path is not None else None,
            "train_config": {
                **{key: value for key, value in asdict(train_config).items() if key != "stages"},
                "stages": [asdict(stage) for stage in train_config.stages],
            },
            "best_step": best_step,
            "best_val_loss": best_metric,
            "test_metrics": test_metrics,
            "model_type": "voxel_patch_mlp_autoencoder",
            "stage_summary": stage_rows,
        },
        checkpoint,
    )
    summary = {
        "run": run.name,
        "checkpoint": checkpoint.as_posix(),
        "metrics": (run / "metrics.csv").as_posix(),
        "stage_summary": (run / "stage_summary.csv").as_posix(),
        "init_checkpoint": init_checkpoint_path.as_posix() if init_checkpoint_path is not None else None,
        "best_step": best_step,
        "best_val_loss": best_metric,
        "duration_s": time.perf_counter() - start_time,
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


__all__ = [
    "VOXEL_AE_SCHEMA_VERSION",
    "VoxelGridConfig",
    "VoxelMLPConfig",
    "VoxelAEStage",
    "VoxelAETrainConfig",
    "VoxelPatchMLPAutoencoder",
    "VoxelSparseCache",
    "build_voxel_energy_cache",
    "corrupt_voxel_batch",
    "make_decay_voxel_curriculum_stages",
    "train_voxel_ae_run",
    "voxel_ae_loss",
    "voxelize_particle_energy",
]

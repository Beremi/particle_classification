from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .edge_training import (
    EdgeDatasetConfig,
    ParticleShard,
    assign_split,
    discover_particle_shards,
    edge_window_manifest_row,
    load_particle_shard,
    make_edge_window,
    summarize_manifest,
    write_edge_window_npz,
)
from .particles import DBSCANParticleParams, write_dict_rows


@dataclass(frozen=True)
class MixedEdgeDatasetConfig:
    windows: int = 4_000
    seed: int = 20260503
    synthetic_fraction: float = 0.35
    hard_fraction: float = 0.45
    min_particles: int = 2
    max_particles: int = 8
    min_particle_hits: int = 4
    max_particle_hits: int = 192
    max_templates: int = 20_000
    max_source_shards: int = 512
    noise_fraction_min: float = 0.02
    noise_fraction_max: float = 0.12
    k_neighbors: int = 16
    radius: float = 4.5
    min_hits: int = 8
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    detector_size: int = 256


@dataclass(frozen=True)
class ParticleTemplate:
    x_centered: np.ndarray
    y_centered: np.ndarray
    t_zeroed: np.ndarray
    energy: np.ndarray
    tot: np.ndarray
    ftoa: np.ndarray
    source_path: str
    source_particle_id: int

    @property
    def n_hits(self) -> int:
        return int(self.x_centered.shape[0])


def build_mixed_edge_training_set(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    params_path: str | Path,
    manifest: str | Path | None = None,
    config: MixedEdgeDatasetConfig | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Create controlled pass-1 windows from shifted real particles and synthetic tracks.

    The output uses the same NPZ/manifest contract as `build_edge_training_set`,
    so the regular EdgeTrackNet trainer can consume it directly.
    """

    config = config or MixedEdgeDatasetConfig()
    output = Path(output_dir)
    windows_dir = output / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    params = DBSCANParticleParams.from_mapping(json.loads(Path(params_path).read_text(encoding="utf-8")))
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    templates = collect_particle_templates(input_path, manifest=manifest, config=config, rng=rng, verbose=verbose)
    if verbose:
        print(f"collected {len(templates)} particle templates for mixed-window generation", flush=True)

    edge_config = EdgeDatasetConfig(
        k_neighbors=config.k_neighbors,
        radius=config.radius,
        min_hits=config.min_hits,
        min_stability_ari=0.0,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    manifest_rows: list[dict[str, object]] = []
    attempts = 0
    while len(manifest_rows) < config.windows and attempts < config.windows * 20:
        attempts += 1
        use_synthetic = not templates or rng.random() < config.synthetic_fraction
        hard = rng.random() < config.hard_fraction
        if use_synthetic:
            window, source_label = make_procedural_window(rng, np_rng, params=params, config=config, hard=hard)
            generator = "procedural"
        else:
            window, source_label = make_mixed_template_window(
                templates,
                rng,
                np_rng,
                params=params,
                config=config,
                hard=hard,
            )
            generator = "mixed_templates"

        edge_window = make_edge_window(
            window,
            params=params,
            config=edge_config,
            stability_ari_override=1.0,
        )
        if edge_window is None:
            continue

        split = assign_split(rng, edge_config)
        window_id = len(manifest_rows)
        rel_name = f"mixed_window_{window_id:07d}.npz"
        path = windows_dir / rel_name
        shard = ParticleShard(path=Path(source_label), source_path=source_label, row_count=int(window["hit_x"].shape[0]))
        write_edge_window_npz(path, edge_window, shard, window_id, split, params)
        row = edge_window_manifest_row(path, edge_window, shard, window_id, split)
        row["generator"] = generator
        row["hard_case"] = int(hard)
        row["component_sources"] = source_label
        manifest_rows.append(row)
        if verbose and len(manifest_rows) % 250 == 0:
            print(f"generated {len(manifest_rows)}/{config.windows} mixed edge windows", flush=True)

    manifest_path = output / "manifest.csv"
    write_dict_rows(manifest_path, manifest_rows)
    summary = {
        "windows_requested": config.windows,
        "attempts": attempts,
        "templates": len(templates),
        **summarize_manifest(manifest_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": manifest_path.as_posix(), **summary}


def collect_particle_templates(
    input_path: str | Path,
    *,
    manifest: str | Path | None,
    config: MixedEdgeDatasetConfig,
    rng: random.Random,
    verbose: bool = False,
) -> list[ParticleTemplate]:
    shards = discover_particle_shards(input_path, manifest)
    rng.shuffle(shards)
    templates: list[ParticleTemplate] = []
    for shard_idx, shard in enumerate(shards[: config.max_source_shards], start=1):
        if len(templates) >= config.max_templates:
            break
        if "suspicious_merged_clusters" in (shard.validation_warnings or ""):
            continue
        try:
            source = load_particle_shard(shard.path)
        except Exception:
            continue
        labels = np.asarray(source["hit_particle_id"], dtype=np.int32)
        for particle_id in sorted(set(int(label) for label in labels.tolist()) - {-1}):
            mask = labels == particle_id
            n_hits = int(np.sum(mask))
            if n_hits < config.min_particle_hits or n_hits > config.max_particle_hits:
                continue
            x = np.asarray(source["hit_x"], dtype=np.float32)[mask]
            y = np.asarray(source["hit_y"], dtype=np.float32)[mask]
            t = np.asarray(source["hit_time"], dtype=np.float64)[mask]
            templates.append(
                ParticleTemplate(
                    x_centered=x - float(np.mean(x)),
                    y_centered=y - float(np.mean(y)),
                    t_zeroed=t - float(np.min(t)),
                    energy=np.asarray(source["hit_energy"], dtype=np.float32)[mask],
                    tot=np.asarray(source["hit_tot"], dtype=np.float32)[mask],
                    ftoa=np.asarray(source["hit_ftoa"], dtype=np.float32)[mask],
                    source_path=shard.source_path,
                    source_particle_id=particle_id,
                )
            )
            if len(templates) >= config.max_templates:
                break
        if verbose and shard_idx % 64 == 0:
            print(f"template scan {shard_idx}/{min(len(shards), config.max_source_shards)}: {len(templates)} templates", flush=True)
    rng.shuffle(templates)
    return templates


def make_mixed_template_window(
    templates: list[ParticleTemplate],
    rng: random.Random,
    np_rng: np.random.Generator,
    *,
    params: DBSCANParticleParams,
    config: MixedEdgeDatasetConfig,
    hard: bool = False,
) -> tuple[dict[str, np.ndarray], str]:
    n_particles = rng.randint(config.min_particles, config.max_particles)
    chosen = [templates[rng.randrange(len(templates))] for _ in range(n_particles)]
    parts = []
    sources = []
    time_cursor = rng.uniform(0.0, 4.0) * params.time_scale
    hard_center_x = rng.uniform(35.0, config.detector_size - 36.0)
    hard_center_y = rng.uniform(35.0, config.detector_size - 36.0)
    hard_time = rng.uniform(0.0, 4.0) * params.time_scale
    for label, template in enumerate(chosen):
        x = template.x_centered.copy()
        y = template.y_centered.copy()
        t = template.t_zeroed.copy()
        if rng.random() < 0.5:
            x = -x
        if rng.random() < 0.5:
            y = -y

        if hard:
            center_x = hard_center_x + rng.uniform(-3.0, 3.0)
            center_y = hard_center_y + rng.uniform(-3.0, 3.0)
        else:
            center_x = rng.uniform(18.0, config.detector_size - 19.0)
            center_y = rng.uniform(18.0, config.detector_size - 19.0)
        x = x + center_x + np_rng.normal(0.0, 0.15, size=x.shape)
        y = y + center_y + np_rng.normal(0.0, 0.15, size=y.shape)
        x, y = shift_inside_detector(x, y, detector_size=config.detector_size)

        if hard:
            t_offset = hard_time + rng.uniform(-0.75, 0.75) * params.time_scale
        else:
            time_cursor += rng.uniform(0.0, 3.0) * params.time_scale
            t_offset = time_cursor
        t = t + t_offset + np_rng.normal(0.0, 0.01 * params.time_scale, size=t.shape)

        parts.append(
            {
                "hit_x": x.astype(np.float32),
                "hit_y": y.astype(np.float32),
                "hit_time": t.astype(np.float64),
                "hit_energy": template.energy.astype(np.float32),
                "hit_tot": template.tot.astype(np.float32),
                "hit_ftoa": template.ftoa.astype(np.float32),
                "hit_particle_id": np.full(template.n_hits, label, dtype=np.int32),
            }
        )
        sources.append(f"{template.source_path}#{template.source_particle_id}")
    window = concatenate_parts(parts)
    add_noise_hits(window, rng, np_rng, params=params, config=config)
    return sort_window_by_time(window), ";".join(sources)


def make_procedural_window(
    rng: random.Random,
    np_rng: np.random.Generator,
    *,
    params: DBSCANParticleParams,
    config: MixedEdgeDatasetConfig,
    hard: bool = False,
) -> tuple[dict[str, np.ndarray], str]:
    parts = []
    n_particles = rng.randint(config.min_particles, config.max_particles)
    time_cursor = rng.uniform(0.0, 2.0) * params.time_scale
    hard_center_x = rng.uniform(35.0, config.detector_size - 36.0)
    hard_center_y = rng.uniform(35.0, config.detector_size - 36.0)
    hard_time = rng.uniform(0.0, 2.0) * params.time_scale
    for label in range(n_particles):
        n_hits = rng.randint(max(5, config.min_particle_hits), min(config.max_particle_hits, 96))
        kind = rng.choice(["line", "kink", "blob", "dense_track"])
        if hard:
            center_x = hard_center_x + rng.uniform(-4.0, 4.0)
            center_y = hard_center_y + rng.uniform(-4.0, 4.0)
        else:
            center_x = rng.uniform(22.0, config.detector_size - 23.0)
            center_y = rng.uniform(22.0, config.detector_size - 23.0)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        length = rng.uniform(4.0, 36.0)
        s = np.linspace(-0.5, 0.5, n_hits, dtype=np.float32)
        if kind == "blob":
            x = center_x + np_rng.normal(0.0, rng.uniform(0.8, 3.0), size=n_hits)
            y = center_y + np_rng.normal(0.0, rng.uniform(0.8, 3.0), size=n_hits)
            t_local = np_rng.normal(0.0, 0.08 * params.time_scale, size=n_hits)
        else:
            bend = np.sin(np.linspace(0.0, np.pi, n_hits)) if kind == "kink" else 0.0
            width = 0.35 if kind == "dense_track" else 0.7
            x = center_x + np.cos(theta) * length * s - np.sin(theta) * bend * length * 0.25
            y = center_y + np.sin(theta) * length * s + np.cos(theta) * bend * length * 0.25
            x = x + np_rng.normal(0.0, width, size=n_hits)
            y = y + np_rng.normal(0.0, width, size=n_hits)
            t_pitch = rng.uniform(0.015, 0.12) * params.time_scale
            t_local = np.arange(n_hits, dtype=np.float64) * t_pitch + np_rng.normal(0.0, 0.01 * params.time_scale, n_hits)
        x, y = shift_inside_detector(x, y, detector_size=config.detector_size)
        if hard:
            t_offset = hard_time + rng.uniform(-0.7, 0.7) * params.time_scale
        else:
            time_cursor += rng.uniform(0.0, 2.5) * params.time_scale
            t_offset = time_cursor
        energy = np_rng.lognormal(mean=1.1, sigma=0.35, size=n_hits).astype(np.float32)
        parts.append(
            {
                "hit_x": x.astype(np.float32),
                "hit_y": y.astype(np.float32),
                "hit_time": (t_offset + t_local).astype(np.float64),
                "hit_energy": energy,
                "hit_tot": np.expm1(energy).astype(np.float32),
                "hit_ftoa": np_rng.uniform(0.0, 30.0, size=n_hits).astype(np.float32),
                "hit_particle_id": np.full(n_hits, label, dtype=np.int32),
            }
        )
    window = concatenate_parts(parts)
    add_noise_hits(window, rng, np_rng, params=params, config=config)
    return sort_window_by_time(window), "procedural"


def shift_inside_detector(x: np.ndarray, y: np.ndarray, *, detector_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    margin = 1.0
    if x.size:
        if float(np.min(x)) < margin:
            x = x + (margin - float(np.min(x)))
        if float(np.max(x)) > detector_size - 1 - margin:
            x = x - (float(np.max(x)) - (detector_size - 1 - margin))
    if y.size:
        if float(np.min(y)) < margin:
            y = y + (margin - float(np.min(y)))
        if float(np.max(y)) > detector_size - 1 - margin:
            y = y - (float(np.max(y)) - (detector_size - 1 - margin))
    return np.clip(x, 0.0, detector_size - 1.0), np.clip(y, 0.0, detector_size - 1.0)


def concatenate_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ["hit_x", "hit_y", "hit_time", "hit_energy", "hit_tot", "hit_ftoa", "hit_particle_id"]
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def add_noise_hits(
    window: dict[str, np.ndarray],
    rng: random.Random,
    np_rng: np.random.Generator,
    *,
    params: DBSCANParticleParams,
    config: MixedEdgeDatasetConfig,
) -> None:
    n_hits = int(window["hit_x"].shape[0])
    noise_fraction = rng.uniform(config.noise_fraction_min, config.noise_fraction_max)
    n_noise = int(round(n_hits * noise_fraction))
    if n_noise <= 0:
        return
    t_min = float(np.min(window["hit_time"])) if n_hits else 0.0
    t_max = float(np.max(window["hit_time"])) if n_hits else params.time_scale
    if t_max <= t_min:
        t_max = t_min + params.time_scale
    energy = np_rng.lognormal(mean=0.7, sigma=0.5, size=n_noise).astype(np.float32)
    noise = {
        "hit_x": np_rng.uniform(0.0, config.detector_size - 1.0, n_noise).astype(np.float32),
        "hit_y": np_rng.uniform(0.0, config.detector_size - 1.0, n_noise).astype(np.float32),
        "hit_time": np_rng.uniform(t_min, t_max, n_noise).astype(np.float64),
        "hit_energy": energy,
        "hit_tot": np.expm1(energy).astype(np.float32),
        "hit_ftoa": np_rng.uniform(0.0, 30.0, n_noise).astype(np.float32),
        "hit_particle_id": np.full(n_noise, -1, dtype=np.int32),
    }
    for key, value in noise.items():
        window[key] = np.concatenate([window[key], value], axis=0)


def sort_window_by_time(window: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.argsort(window["hit_time"], kind="mergesort")
    return {key: np.asarray(value)[order] for key, value in window.items()}

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from particle_classification.dbscan.pipeline import write_dict_rows
from particle_classification.experiments.autoencoders.voxel import VoxelGridConfig, flush_voxel_chunk


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def iter_ok_rows(manifest: Path):
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            if row.get("status") == "ok":
                yield row_index, row


class ChunkCache:
    def __init__(self, max_chunks: int = 8):
        self.max_chunks = max_chunks
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def load(self, path: str) -> dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        with np.load(path, allow_pickle=False) as data:
            chunk = {key: data[key] for key in data.files}
        self._cache[path] = chunk
        while len(self._cache) > self.max_chunks:
            self._cache.popitem(last=False)
        return chunk


def compressed_sparse_from_row(
    row: dict[str, str],
    chunk_cache: ChunkCache,
    *,
    src_grid: VoxelGridConfig,
    dst_grid: VoxelGridConfig,
    mirror_x: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    chunk = chunk_cache.load(row["chunk_path"])
    chunk_row = int(row["chunk_row"])
    start = int(chunk["voxel_offsets"][chunk_row])
    end = int(chunk["voxel_offsets"][chunk_row + 1])
    if end <= start:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    flat = chunk["voxel_index"][start:end].astype(np.int64)
    values = chunk["voxel_value"][start:end].astype(np.float32)
    src_xy = src_grid.y_bins * src_grid.x_bins
    t = flat // src_xy
    rem = flat - t * src_xy
    y = rem // src_grid.x_bins
    x = rem - y * src_grid.x_bins
    t_factor = src_grid.t_bins // dst_grid.t_bins
    y_factor = src_grid.y_bins // dst_grid.y_bins
    x_factor = src_grid.x_bins // dst_grid.x_bins
    ct = np.clip(t // t_factor, 0, dst_grid.t_bins - 1)
    cy = np.clip(y // y_factor, 0, dst_grid.y_bins - 1)
    cx = np.clip(x // x_factor, 0, dst_grid.x_bins - 1)
    if mirror_x:
        cx = dst_grid.x_bins - 1 - cx
    cflat = (ct * dst_grid.y_bins * dst_grid.x_bins + cy * dst_grid.x_bins + cx).astype(np.int64)
    unique, inverse = np.unique(cflat, return_inverse=True)
    summed = np.zeros(unique.shape[0], dtype=np.float32)
    np.add.at(summed, inverse, values)
    return unique.astype(np.int32), summed.astype(np.float32)


def hit_bucket(n_hits: int) -> str:
    if n_hits <= 4:
        return "2-4"
    if n_hits <= 10:
        return "5-10"
    if n_hits <= 50:
        return "11-50"
    return "51+"


def make_projection(dst_grid: VoxelGridConfig, projection_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((projection_dim, dst_grid.flat_dim), dtype=np.float32)
    projection /= np.sqrt(float(projection_dim))
    return projection


def representative_group(
    row: dict[str, str],
    indices: np.ndarray,
    values: np.ndarray,
    projection: np.ndarray,
    *,
    projection_bits: int,
) -> tuple[str, int]:
    energy = float(np.sum(values))
    density = values / max(energy, 1e-8)
    features = projection[:, indices].dot(density.astype(np.float32)) if indices.size else np.zeros(projection.shape[0], dtype=np.float32)
    signature = 0
    for bit, value in enumerate(features[:projection_bits]):
        if float(value) >= 0.0:
            signature |= 1 << bit
    n_hits = int(float(row.get("n_hits", 0)))
    occupancy = int(indices.shape[0])
    occupancy_bucket = min(7, int(math.log2(max(occupancy, 1))))
    key = f"{row.get('split')}|{hit_bucket(n_hits)}|occ{occupancy_bucket}|sig{signature:0{max(1, projection_bits // 4)}x}"
    return key, signature


def allocate_targets(group_counts: dict[str, int], target: int) -> dict[str, int]:
    if target <= 0 or not group_counts:
        return {}
    target = min(target, sum(group_counts.values()))
    groups = sorted(group_counts, key=lambda key: stable_int(key))
    quotas = {group: 0 for group in groups}
    remaining = target
    while remaining > 0:
        progressed = False
        for group in groups:
            if quotas[group] >= group_counts[group]:
                continue
            quotas[group] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def should_select_evenly(seen_before: int, group_count: int, quota: int) -> bool:
    if quota <= 0:
        return False
    return ((seen_before + 1) * quota // group_count) > (seen_before * quota // group_count)


def make_view(row_index: int, row: dict[str, str], indices: np.ndarray, values: np.ndarray, group: str, signature: int, *, mirror_axis: str | None) -> dict[str, object]:
    return {
        "voxel_index": indices,
        "voxel_value": values,
        "status": "ok",
        "split": row["split"],
        "split_group": row["split_group"],
        "source_npz": row["source_npz"],
        "source_path": row["source_path"],
        "particle_id": int(float(row["particle_id"])),
        "particle_index": int(float(row["particle_index"])),
        "n_hits": int(float(row["n_hits"])),
        "kept_hits": int(float(row.get("kept_hits", row["n_hits"]))),
        "kept_fraction": float(row.get("kept_fraction", 1.0)),
        "kept_energy_fraction": float(row.get("kept_energy_fraction", 1.0)),
        "occupied_voxels": int(indices.shape[0]),
        "x_span": float(row.get("x_span", 0.0)),
        "y_span": float(row.get("y_span", 0.0)),
        "time_span": float(row.get("time_span", 0.0)),
        "representative_group": group,
        "representative_signature": signature,
        "representative_source_row": row_index,
        "mirror_axis": mirror_axis or "",
    }


def flush_pending(pending: list[dict[str, object]], output: Path, chunk_id: int, config: VoxelGridConfig, rows: list[dict[str, object]]) -> int:
    if not pending:
        return chunk_id
    rows.extend(flush_voxel_chunk(pending, output / "chunks", chunk_id, config))
    pending.clear()
    return chunk_id + 1


def build_compressed_subset(
    source_cache: Path,
    output: Path,
    mirror_output: Path,
    *,
    target_particles: int,
    seed: int,
    projection_dim: int,
    projection_bits: int,
    chunk_size: int,
    verbose: bool,
) -> dict[str, object]:
    source_summary = json.loads((source_cache / "summary.json").read_text(encoding="utf-8"))
    src_grid = VoxelGridConfig(**source_summary["grid_config"])
    dst_grid = VoxelGridConfig(
        t_bins=src_grid.t_bins // 4,
        y_bins=src_grid.y_bins // 2,
        x_bins=src_grid.x_bins // 2,
        time_bin=src_grid.time_bin * 4,
        xy_bin=src_grid.xy_bin * 2,
        energy_norm=src_grid.energy_norm,
        min_particle_hits=src_grid.min_particle_hits,
        seed=seed,
        val_fraction=src_grid.val_fraction,
        test_fraction=src_grid.test_fraction,
        chunk_size=chunk_size,
    )
    output.mkdir(parents=True, exist_ok=True)
    mirror_output.mkdir(parents=True, exist_ok=True)
    (output / "chunks").mkdir(exist_ok=True)
    (mirror_output / "chunks").mkdir(exist_ok=True)
    projection = make_projection(dst_grid, projection_dim, seed)
    manifest = source_cache / "manifest.csv"

    group_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    cache = ChunkCache()
    t0 = time.perf_counter()
    for seen, (row_index, row) in enumerate(iter_ok_rows(manifest), start=1):
        indices, values = compressed_sparse_from_row(row, cache, src_grid=src_grid, dst_grid=dst_grid)
        group, _ = representative_group(row, indices, values, projection, projection_bits=projection_bits)
        group_counts[group] += 1
        split_counts[row["split"]] += 1
        if verbose and seen % 100_000 == 0:
            print(f"pass1 {seen:,} rows, {len(group_counts):,} groups", flush=True)

    total_rows = sum(split_counts.values())
    split_targets = {split: int(round(target_particles * count / max(total_rows, 1))) for split, count in split_counts.items()}
    diff = min(target_particles, total_rows) - sum(split_targets.values())
    for split in sorted(split_targets, key=lambda item: -split_counts[item]):
        if diff == 0:
            break
        split_targets[split] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
    counts_by_split: dict[str, dict[str, int]] = defaultdict(dict)
    for group, count in group_counts.items():
        split = group.split("|", 1)[0]
        counts_by_split[split][group] = count
    quotas: dict[str, int] = {}
    for split, counts in counts_by_split.items():
        quotas.update(allocate_targets(counts, split_targets.get(split, 0)))

    selected_rows: list[dict[str, object]] = []
    mirror_rows: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    mirror_pending: list[dict[str, object]] = []
    chunk_id = 0
    mirror_chunk_id = 0
    seen_by_group: dict[str, int] = defaultdict(int)
    selected_by_group: dict[str, int] = defaultdict(int)
    cache = ChunkCache()
    for seen, (row_index, row) in enumerate(iter_ok_rows(manifest), start=1):
        indices, values = compressed_sparse_from_row(row, cache, src_grid=src_grid, dst_grid=dst_grid)
        group, signature = representative_group(row, indices, values, projection, projection_bits=projection_bits)
        seen_before = seen_by_group[group]
        seen_by_group[group] += 1
        quota = quotas.get(group, 0)
        if not should_select_evenly(seen_before, group_counts[group], quota):
            continue
        selected_by_group[group] += 1
        pending.append(make_view(row_index, row, indices, values, group, signature, mirror_axis=None))
        mirror_indices, mirror_values = compressed_sparse_from_row(row, cache, src_grid=src_grid, dst_grid=dst_grid, mirror_x=True)
        mirror_pending.append(make_view(row_index, row, mirror_indices, mirror_values, group, signature, mirror_axis="x"))
        if len(pending) >= chunk_size:
            chunk_id = flush_pending(pending, output, chunk_id, dst_grid, selected_rows)
            mirror_chunk_id = flush_pending(mirror_pending, mirror_output, mirror_chunk_id, dst_grid, mirror_rows)
        if verbose and seen % 100_000 == 0:
            print(f"pass2 {seen:,} rows, selected {len(selected_rows) + len(pending):,}", flush=True)
    chunk_id = flush_pending(pending, output, chunk_id, dst_grid, selected_rows)
    mirror_chunk_id = flush_pending(mirror_pending, mirror_output, mirror_chunk_id, dst_grid, mirror_rows)

    write_dict_rows(output / "manifest.csv", selected_rows)
    write_dict_rows(mirror_output / "manifest.csv", mirror_rows)
    summary = {
        "schema_version": "phase2_compressed_representative_v1",
        "source_cache": source_cache.as_posix(),
        "output": output.as_posix(),
        "mirror_output": mirror_output.as_posix(),
        "target_particles": target_particles,
        "selected_particles": len(selected_rows),
        "selected_mirror_particles": len(mirror_rows),
        "source_particles": total_rows,
        "source_split_counts": dict(split_counts),
        "selected_splits": {split: sum(1 for row in selected_rows if row.get("split") == split) for split in ["train", "val", "test"]},
        "group_count": len(group_counts),
        "projection_dim": projection_dim,
        "projection_bits": projection_bits,
        "src_grid_config": asdict(src_grid),
        "grid_config": asdict(dst_grid),
        "duration_s": time.perf_counter() - t0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    mirror_summary = {**summary, "output": mirror_output.as_posix(), "mirror_axis": "x", "manifest": (mirror_output / "manifest.csv").as_posix()}
    (mirror_output / "summary.json").write_text(json.dumps(mirror_summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an 8x32x32 representative compressed voxel cache and an x-mirrored copy.")
    parser.add_argument("--source-cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_32x64x64_curriculum_v001"))
    parser.add_argument("--out", type=Path, default=Path("local_data/processed/phase2_voxel_energy_8x32x32_representative_v001"))
    parser.add_argument("--mirror-out", type=Path, default=Path("local_data/processed/phase2_voxel_energy_8x32x32_representative_mirror_x_v001"))
    parser.add_argument("--target-particles", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--projection-dim", type=int, default=24)
    parser.add_argument("--projection-bits", type=int, default=18)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    summary = build_compressed_subset(
        args.source_cache,
        args.out,
        args.mirror_out,
        target_particles=args.target_particles,
        seed=args.seed,
        projection_dim=args.projection_dim,
        projection_bits=args.projection_bits,
        chunk_size=args.chunk_size,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

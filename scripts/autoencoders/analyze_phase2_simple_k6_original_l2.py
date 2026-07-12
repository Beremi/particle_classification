#!/usr/bin/env python3
"""Check K=6 latent groups against original voxel-tensor L2 distances."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

from particle_classification.experiments.autoencoders.voxel import VoxelGridConfig, load_voxel_manifest


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ChunkCache:
    def __init__(self, max_items: int = 24):
        self.max_items = max_items
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def load(self, path: str) -> dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        with np.load(path, allow_pickle=False) as data:
            chunk = {key: data[key] for key in data.files}
        self._cache[path] = chunk
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return chunk


def load_rows(cache_dir: Path) -> list[dict[str, str]]:
    manifest = cache_dir / "manifest.csv"
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        split_rows = load_voxel_manifest(manifest, split=split, max_items=None)
        for row in split_rows:
            row["split"] = split
        rows.extend(split_rows)
    return rows


def select_samples(labels: np.ndarray, per_group: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for group in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == group)
        count = min(per_group, idx.size)
        selected.append(rng.choice(idx, size=count, replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def row_sparse_values(row: dict[str, str], cache: ChunkCache) -> tuple[np.ndarray, np.ndarray]:
    chunk = cache.load(row["chunk_path"])
    chunk_row = int(row["chunk_row"])
    start = int(chunk["voxel_offsets"][chunk_row])
    end = int(chunk["voxel_offsets"][chunk_row + 1])
    return chunk["voxel_index"][start:end].astype(np.int64, copy=False), chunk["voxel_value"][start:end].astype(np.float32, copy=False)


def scatter_dense(dense: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
    if indices.size:
        np.add.at(dense, indices, values)


def compute_exact_and_samples(
    *,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    grid: VoxelGridConfig,
    sample_indices: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, str]]]:
    group_to_pos = {int(group): pos for pos, group in enumerate(groups)}
    sample_lookup = {int(global_idx): pos for pos, global_idx in enumerate(sample_indices)}
    sample_dense = np.zeros((len(sample_indices), grid.flat_dim), dtype=np.float32)
    sample_rows: list[dict[str, str]] = [{} for _ in range(len(sample_indices))]

    sums = np.zeros((len(groups), grid.flat_dim), dtype=np.float64)
    sums_unit = np.zeros_like(sums)
    counts = np.zeros(len(groups), dtype=np.int64)
    sum_norm2 = np.zeros(len(groups), dtype=np.float64)
    sum_unit_norm2 = np.zeros(len(groups), dtype=np.float64)
    energy_sum = np.zeros(len(groups), dtype=np.float64)

    cache = ChunkCache()
    for global_idx, row in enumerate(rows):
        group_pos = group_to_pos[int(labels[global_idx])]
        indices, values = row_sparse_values(row, cache)
        counts[group_pos] += 1
        if values.size:
            np.add.at(sums[group_pos], indices, values.astype(np.float64))
            norm2 = float(np.dot(values, values))
            sum_norm2[group_pos] += norm2
            energy = float(values.sum())
            energy_sum[group_pos] += energy
            if energy > 1e-12:
                unit_values = values.astype(np.float64) / energy
                np.add.at(sums_unit[group_pos], indices, unit_values)
                sum_unit_norm2[group_pos] += float(np.dot(unit_values, unit_values))
        sample_pos = sample_lookup.get(global_idx)
        if sample_pos is not None:
            scatter_dense(sample_dense[sample_pos], indices, values)
            sample_rows[sample_pos] = row
        if global_idx and global_idx % 200_000 == 0:
            print(f"streamed {global_idx:,}/{len(rows):,} original tensors", flush=True)

    return (
        {
            "sums": sums,
            "sums_unit": sums_unit,
            "counts": counts,
            "sum_norm2": sum_norm2,
            "sum_unit_norm2": sum_unit_norm2,
            "energy_sum": energy_sum,
        },
        sample_dense,
        sample_rows,
    )


def centroid_stats(exact: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    counts = exact["counts"].astype(np.float64)
    centroids = exact["sums"] / np.maximum(counts[:, None], 1.0)
    centroids_unit = exact["sums_unit"] / np.maximum(counts[:, None], 1.0)
    mean_norm2 = exact["sum_norm2"] / np.maximum(counts, 1.0)
    mean_unit_norm2 = exact["sum_unit_norm2"] / np.maximum(counts, 1.0)
    rms = np.sqrt(np.maximum(mean_norm2 - np.sum(centroids * centroids, axis=1), 0.0))
    rms_unit = np.sqrt(np.maximum(mean_unit_norm2 - np.sum(centroids_unit * centroids_unit, axis=1), 0.0))
    centroid_dist = pairwise_l2_matrix(centroids)
    centroid_dist_unit = pairwise_l2_matrix(centroids_unit)
    return {
        "centroids": centroids,
        "centroids_unit": centroids_unit,
        "rms": rms,
        "rms_unit": rms_unit,
        "centroid_dist": centroid_dist,
        "centroid_dist_unit": centroid_dist_unit,
    }


def pairwise_l2_matrix(x: np.ndarray) -> np.ndarray:
    norm2 = np.sum(x * x, axis=1)
    dist2 = norm2[:, None] + norm2[None, :] - 2.0 * (x @ x.T)
    return np.sqrt(np.maximum(dist2, 0.0))


def random_pair_distances(a: np.ndarray, b: np.ndarray, n_pairs: int, rng: np.random.Generator, same: bool) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.asarray([], dtype=np.float32)
    out = np.empty(n_pairs, dtype=np.float32)
    written = 0
    chunk = 8192
    while written < n_pairs:
        size = min(chunk, n_pairs - written)
        ia = rng.integers(0, len(a), size=size)
        ib = rng.integers(0, len(b), size=size)
        if same and len(a) > 1:
            equal = ia == ib
            while np.any(equal):
                ib[equal] = rng.integers(0, len(b), size=int(equal.sum()))
                equal = ia == ib
        diff = a[ia].astype(np.float32, copy=False) - b[ib].astype(np.float32, copy=False)
        out[written : written + size] = np.sqrt(np.sum(diff * diff, axis=1))
        written += size
    return out


def summarize_distances(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"mean": float(values.mean()), "p10": float(p10), "p50": float(p50), "p90": float(p90)}


def unit_sum_dense(x: np.ndarray) -> np.ndarray:
    denom = np.sum(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def sampled_pair_stats(
    sample_dense: np.ndarray,
    sample_labels: np.ndarray,
    groups: np.ndarray,
    n_pairs: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    sample_unit = unit_sum_dense(sample_dense)
    within_rows: list[dict[str, object]] = []
    between_rows: list[dict[str, object]] = []
    by_group = {int(group): np.flatnonzero(sample_labels == group) for group in groups}

    for group in groups:
        idx = by_group[int(group)]
        raw = random_pair_distances(sample_dense[idx], sample_dense[idx], n_pairs, rng, same=True)
        unit = random_pair_distances(sample_unit[idx], sample_unit[idx], n_pairs, rng, same=True)
        raw_stats = summarize_distances(raw)
        unit_stats = summarize_distances(unit)
        within_rows.append(
            {
                "group": int(group),
                "sample_count": int(idx.size),
                "within_raw_mean": raw_stats["mean"],
                "within_raw_p10": raw_stats["p10"],
                "within_raw_p50": raw_stats["p50"],
                "within_raw_p90": raw_stats["p90"],
                "within_unit_mean": unit_stats["mean"],
                "within_unit_p10": unit_stats["p10"],
                "within_unit_p50": unit_stats["p50"],
                "within_unit_p90": unit_stats["p90"],
            }
        )

    for pos_i, group_i in enumerate(groups):
        for group_j in groups[pos_i + 1 :]:
            idx_i = by_group[int(group_i)]
            idx_j = by_group[int(group_j)]
            raw = random_pair_distances(sample_dense[idx_i], sample_dense[idx_j], n_pairs, rng, same=False)
            unit = random_pair_distances(sample_unit[idx_i], sample_unit[idx_j], n_pairs, rng, same=False)
            raw_stats = summarize_distances(raw)
            unit_stats = summarize_distances(unit)
            between_rows.append(
                {
                    "group_i": int(group_i),
                    "group_j": int(group_j),
                    "between_raw_mean": raw_stats["mean"],
                    "between_raw_p10": raw_stats["p10"],
                    "between_raw_p50": raw_stats["p50"],
                    "between_raw_p90": raw_stats["p90"],
                    "between_unit_mean": unit_stats["mean"],
                    "between_unit_p10": unit_stats["p10"],
                    "between_unit_p50": unit_stats["p50"],
                    "between_unit_p90": unit_stats["p90"],
                }
            )
    return within_rows, between_rows


def sample_silhouette(
    sample_dense: np.ndarray,
    sample_labels: np.ndarray,
    max_count: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(sample_dense.shape[0])
    if idx.size > max_count:
        idx = rng.choice(idx, size=max_count, replace=False)
    raw = sample_dense[idx].astype(np.float32, copy=False)
    unit = unit_sum_dense(raw)
    labels = sample_labels[idx]
    return {
        "raw_silhouette": float(silhouette_score(raw, labels, metric="euclidean")),
        "unit_sum_silhouette": float(silhouette_score(unit, labels, metric="euclidean")),
        "sample_count": int(idx.size),
    }


def plot_matrix(matrix: np.ndarray, groups: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=170)
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(groups)), [str(int(g)) for g in groups])
    ax.set_yticks(range(len(groups)), [str(int(g)) for g in groups])
    ax.set_xlabel("group")
    ax.set_ylabel("group")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_within_between(within_rows: list[dict[str, object]], between_rows: list[dict[str, object]], metric_prefix: str, path: Path) -> None:
    groups = [int(row["group"]) for row in within_rows]
    within_p50 = [float(row[f"within_{metric_prefix}_p50"]) for row in within_rows]
    between_p50 = []
    for group in groups:
        vals = [
            float(row[f"between_{metric_prefix}_p50"])
            for row in between_rows
            if int(row["group_i"]) == group or int(row["group_j"]) == group
        ]
        between_p50.append(float(np.min(vals)) if vals else float("nan"))
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    x = np.arange(len(groups))
    width = 0.38
    ax.bar(x - width / 2, within_p50, width, label="within group p50")
    ax.bar(x + width / 2, between_p50, width, label="nearest between-group p50")
    ax.set_xticks(x, [str(g) for g in groups])
    ax.set_xlabel("K=6 group")
    ax.set_ylabel("sampled L2 distance")
    ax.set_title(f"Original tensor {metric_prefix} L2: within vs nearest between")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_8x32x32_representative_v001"))
    parser.add_argument(
        "--k6-labels",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_k6_group_histograms_v001/kmeans_k6_labels.npz"),
    )
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/latent_analysis/phase2-simple-z8-k6-original-l2-diagnostics.md"))
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("experimental_notes/assets/phase2_simple_z8_k6_original_l2_diagnostics_v001"),
    )
    parser.add_argument(
        "--local-out",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_k6_original_l2_diagnostics_v001"),
    )
    parser.add_argument("--sample-per-group", type=int, default=1500)
    parser.add_argument("--pairs-per-estimate", type=int, default=50000)
    parser.add_argument("--silhouette-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    ensure_dir(args.out.parent)
    ensure_dir(args.asset_dir)
    ensure_dir(args.local_out)

    summary_payload = json.loads((args.cache / "summary.json").read_text(encoding="utf-8"))
    grid = VoxelGridConfig(**summary_payload["grid_config"])
    labels = np.load(args.k6_labels, allow_pickle=False)["labels"].astype(np.int32)
    rows = load_rows(args.cache)
    if len(rows) != labels.shape[0]:
        raise ValueError(f"row/label mismatch: {len(rows)} rows vs {labels.shape[0]} labels")

    groups = np.asarray(sorted(np.unique(labels)), dtype=np.int32)
    sample_indices = select_samples(labels, args.sample_per_group, args.seed)
    exact, sample_dense, sample_rows = compute_exact_and_samples(
        rows=rows,
        labels=labels,
        grid=grid,
        sample_indices=sample_indices,
        groups=groups,
    )
    sample_labels = labels[sample_indices]
    stats = centroid_stats(exact)
    within_rows, between_rows = sampled_pair_stats(
        sample_dense,
        sample_labels,
        groups,
        n_pairs=args.pairs_per_estimate,
        seed=args.seed + 17,
    )
    sil = sample_silhouette(sample_dense, sample_labels, max_count=args.silhouette_sample, seed=args.seed + 23)

    group_rows: list[dict[str, object]] = []
    for pos, group in enumerate(groups):
        nearest_raw = float(np.min(np.delete(stats["centroid_dist"][pos], pos)))
        nearest_unit = float(np.min(np.delete(stats["centroid_dist_unit"][pos], pos)))
        within = within_rows[pos]
        group_rows.append(
            {
                "group": int(group),
                "count": int(exact["counts"][pos]),
                "rms_to_centroid_raw": float(stats["rms"][pos]),
                "nearest_centroid_raw": nearest_raw,
                "centroid_separation_ratio_raw": nearest_raw / max(float(stats["rms"][pos]), 1e-12),
                "sample_within_p50_raw": float(within["within_raw_p50"]),
                "rms_to_centroid_unit_sum": float(stats["rms_unit"][pos]),
                "nearest_centroid_unit_sum": nearest_unit,
                "centroid_separation_ratio_unit_sum": nearest_unit / max(float(stats["rms_unit"][pos]), 1e-12),
                "sample_within_p50_unit_sum": float(within["within_unit_p50"]),
                "mean_energy_sum": float(exact["energy_sum"][pos] / max(exact["counts"][pos], 1)),
            }
        )

    write_csv(
        args.local_out / "group_original_l2_summary.csv",
        group_rows,
        [
            "group",
            "count",
            "rms_to_centroid_raw",
            "nearest_centroid_raw",
            "centroid_separation_ratio_raw",
            "sample_within_p50_raw",
            "rms_to_centroid_unit_sum",
            "nearest_centroid_unit_sum",
            "centroid_separation_ratio_unit_sum",
            "sample_within_p50_unit_sum",
            "mean_energy_sum",
        ],
    )
    write_csv(
        args.local_out / "sampled_within_group_l2.csv",
        within_rows,
        [
            "group",
            "sample_count",
            "within_raw_mean",
            "within_raw_p10",
            "within_raw_p50",
            "within_raw_p90",
            "within_unit_mean",
            "within_unit_p10",
            "within_unit_p50",
            "within_unit_p90",
        ],
    )
    write_csv(
        args.local_out / "sampled_between_group_l2.csv",
        between_rows,
        [
            "group_i",
            "group_j",
            "between_raw_mean",
            "between_raw_p10",
            "between_raw_p50",
            "between_raw_p90",
            "between_unit_mean",
            "between_unit_p10",
            "between_unit_p50",
            "between_unit_p90",
        ],
    )
    np.savez_compressed(
        args.local_out / "sampled_original_tensors_and_labels.npz",
        sample_indices=sample_indices,
        sample_labels=sample_labels,
        sample_dense=sample_dense.astype(np.float16),
    )

    raw_centroid_plot = args.asset_dir / "raw_centroid_l2_matrix.png"
    unit_centroid_plot = args.asset_dir / "unit_sum_centroid_l2_matrix.png"
    raw_bar_plot = args.asset_dir / "raw_within_vs_between_l2.png"
    unit_bar_plot = args.asset_dir / "unit_sum_within_vs_between_l2.png"
    plot_matrix(stats["centroid_dist"], groups, "Raw original tensor centroid L2", raw_centroid_plot)
    plot_matrix(stats["centroid_dist_unit"], groups, "Unit-sum original tensor centroid L2", unit_centroid_plot)
    plot_within_between(within_rows, between_rows, "raw", raw_bar_plot)
    plot_within_between(within_rows, between_rows, "unit", unit_bar_plot)

    rel = lambda p: p.relative_to(args.out.parent).as_posix()
    lines = [
        "# Simple z8 K=6 Original-Tensor L2 Diagnostics",
        "",
        "This checks whether the latent K=6 groups are also compact in the original voxel tensor space. "
        "Distances here are between the original `[8,32,32]` energy tensors, not between reconstructions.",
        "",
        "Two variants are reported:",
        "",
        "- `raw`: direct L2 on stored voxel energy tensors, so total energy/occupancy affects the distance.",
        "- `unit-sum`: each particle tensor is divided by its total voxel energy before L2, so it is closer to a shape-only check.",
        "",
        "## Overall Sample Silhouette In Original Tensor Space",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| sampled particles | {sil['sample_count']:,} |",
        f"| raw original tensor silhouette | {sil['raw_silhouette']:.4f} |",
        f"| unit-sum original tensor silhouette | {sil['unit_sum_silhouette']:.4f} |",
        "",
        "A positive silhouette means the groups have some separation in original-tensor L2 space. "
        "Values near zero mean the groups overlap strongly in that metric.",
        "",
        "## Group Compactness",
        "",
        "| group | count | raw RMS to centroid | raw nearest centroid | raw sep ratio | raw within p50 | unit RMS | unit nearest centroid | unit sep ratio | unit within p50 | mean energy sum |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['group']} | {int(row['count']):,} | {float(row['rms_to_centroid_raw']):.4f} | "
            f"{float(row['nearest_centroid_raw']):.4f} | {float(row['centroid_separation_ratio_raw']):.3f} | "
            f"{float(row['sample_within_p50_raw']):.4f} | {float(row['rms_to_centroid_unit_sum']):.5f} | "
            f"{float(row['nearest_centroid_unit_sum']):.5f} | {float(row['centroid_separation_ratio_unit_sum']):.3f} | "
            f"{float(row['sample_within_p50_unit_sum']):.5f} | {float(row['mean_energy_sum']):.4f} |"
        )
    lines += [
        "",
        "## Distance Plots",
        "",
        f"![raw centroid L2 matrix]({rel(raw_centroid_plot)})",
        "",
        f"![unit-sum centroid L2 matrix]({rel(unit_centroid_plot)})",
        "",
        f"![raw within vs between L2]({rel(raw_bar_plot)})",
        "",
        f"![unit-sum within vs between L2]({rel(unit_bar_plot)})",
        "",
        "## Interpretation",
        "",
        "- If a group's nearest-centroid distance is smaller than its RMS-to-centroid, the group is broader than its separation from another group.",
        "- If sampled within-group p50 is close to nearest between-group p50, the grouping is weak under original-tensor L2.",
        "- Raw L2 is energy-sensitive; unit-sum L2 is a better check for morphology-only grouping.",
        "",
        "## Local Artifacts",
        "",
        f"- group summary: `{(args.local_out / 'group_original_l2_summary.csv').as_posix()}`",
        f"- sampled within distances: `{(args.local_out / 'sampled_within_group_l2.csv').as_posix()}`",
        f"- sampled between distances: `{(args.local_out / 'sampled_between_group_l2.csv').as_posix()}`",
        f"- sampled tensors: `{(args.local_out / 'sampled_original_tensors_and_labels.npz').as_posix()}`",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

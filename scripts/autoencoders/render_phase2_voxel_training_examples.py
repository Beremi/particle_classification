#!/usr/bin/env python3
"""Render clean and stage-1 corrupted voxel energy tensors as 2D time slices."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from particle_classification.experiments.autoencoders.voxel import VoxelGridConfig, corrupt_voxel_batch, voxelize_particle_energy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-manifest", type=Path, default=Path("local_data/processed/phase2_particles_eps5_v001/manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experimental_notes/assets/phase2_voxel_autoencoder/training_stage_examples"))
    parser.add_argument("--markdown", type=Path, default=Path("experimental_notes/autoencoders/voxel/baseline/phase2-voxel-training-examples.md"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=50)
    parser.add_argument("--max-hits", type=int, default=None)
    parser.add_argument("--exact-hits", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260507)
    return parser.parse_args()


def choose_manifest_rows(
    manifest: Path,
    *,
    count: int,
    min_hits: int,
    max_hits: int | None,
    exact_hits: int | None,
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok" or int(float(row.get("view_id", 0))) != 0:
                continue
            n_hits = int(float(row.get("n_hits", 0)))
            if exact_hits is not None:
                if n_hits == exact_hits:
                    rows.append(row)
                continue
            if n_hits < min_hits:
                continue
            if max_hits is not None and n_hits > max_hits:
                continue
            rows.append(row)
    rng = np.random.default_rng(seed)
    if len(rows) <= count:
        return rows
    # Bias toward particles with enough structure to make the slice panels useful.
    weights = np.asarray([min(int(float(row["n_hits"])), 300) for row in rows], dtype=np.float64)
    weights = weights / weights.sum()
    selected = rng.choice(np.arange(len(rows)), size=count, replace=False, p=weights)
    return [rows[int(idx)] for idx in selected]


def dense_from_particle_row(row: dict[str, str], config: VoxelGridConfig) -> tuple[np.ndarray, dict[str, float]]:
    source_npz = Path(row["source_npz"])
    particle_index = int(float(row["particle_index"]))
    with np.load(source_npz, allow_pickle=False) as data:
        offsets = data["particle_offsets"].astype(np.int64)
        start = int(offsets[particle_index])
        end = int(offsets[particle_index + 1])
        voxel_index, voxel_value, metadata = voxelize_particle_energy(
            data["hit_x"][start:end].astype(np.float64),
            data["hit_y"][start:end].astype(np.float64),
            data["hit_time"][start:end].astype(np.float64),
            data["hit_energy"][start:end].astype(np.float64),
            config,
        )
    dense = np.zeros(config.flat_dim, dtype=np.float32)
    dense[voxel_index] = voxel_value
    return dense.reshape(config.t_bins, config.y_bins, config.x_bins), metadata


def time_slice_images(volume: np.ndarray, *, slices: int = 4) -> np.ndarray:
    """Sum a ``[T, Y, X]`` tensor into a few coarse time blocks."""

    parts = np.array_split(np.asarray(volume, dtype=np.float32), slices, axis=0)
    return np.stack([part.sum(axis=0) for part in parts], axis=0)


def render_example(
    row: dict[str, str],
    clean: np.ndarray,
    corrupted: np.ndarray,
    output: Path,
    *,
    rank: int,
    metadata: dict[str, float],
) -> dict[str, object]:
    clean_slices = time_slice_images(clean, slices=4)
    corrupted_slices = time_slice_images(corrupted, slices=4)
    vmax = max(float(clean_slices.max()), float(corrupted_slices.max()), 1e-8)
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=160, constrained_layout=True)
    cmap = "viridis"
    for idx in range(4):
        ax = axes[0, idx]
        ax.imshow(clean_slices[idx], origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"clean t{idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, idx]
        ax.imshow(corrupted_slices[idx], origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"stage-1 t{idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel("clean")
    axes[1, 0].set_ylabel("blurred/noised")
    fig.suptitle(
        f"Example {rank}: {row['source_path']} particle {row['particle_id']} | hits={int(float(row['n_hits']))}",
        fontsize=13,
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=axes.ravel().tolist(), shrink=0.78, pad=0.01)
    cbar.set_label("summed normalized log-ToT energy per 8 time bins")
    fig.savefig(output)
    plt.close(fig)
    return {
        "rank": rank,
        "source_path": row["source_path"],
        "particle_id": row["particle_id"],
        "n_hits": int(float(row["n_hits"])),
        "kept_fraction": metadata["kept_fraction"],
        "kept_energy_fraction": metadata["kept_energy_fraction"],
        "clean_nonzero_pixels": int(np.count_nonzero(clean_slices)),
        "corrupted_nonzero_pixels": int(np.count_nonzero(corrupted_slices)),
        "image": output.as_posix(),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = VoxelGridConfig(t_bins=32, y_bins=64, x_bins=64, time_bin=0.625)
    rows = choose_manifest_rows(
        args.phase2_manifest,
        count=args.count,
        min_hits=args.min_hits,
        max_hits=args.max_hits,
        exact_hits=args.exact_hits,
        seed=args.seed,
    )
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(args.seed)
    report_rows: list[dict[str, object]] = []
    for rank, row in enumerate(rows, start=1):
        clean, metadata = dense_from_particle_row(row, config)
        clean_tensor = torch.from_numpy(clean[None])
        corrupted = corrupt_voxel_batch(clean_tensor, blur_kernel=5, noise_std=0.04, voxel_dropout=0.08)[0].numpy()
        output = args.out_dir / f"voxel_training_example_{rank:02d}.png"
        report_rows.append(render_example(row, clean, corrupted, output, rank=rank, metadata=metadata))
    torch.random.set_rng_state(rng_state)

    csv_path = args.out_dir / "voxel_training_examples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        if report_rows:
            writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)

    lines = [
        "# Phase 2 Voxel Training Examples",
        "",
        "These are real estimated particles converted into logical 3D energy tensors of shape `32 x 64 x 64` (`time x y x x`).",
        "",
        "Each image is a 2D view of the 3D tensor. The 32 time bins are split into four blocks of 8 bins, and each block is summed into one XY image.",
        "",
        "Top row: clean centered energy tensor. Bottom row: the first curriculum-stage training input after `5 x 5 x 5` blur, support noise `0.04`, and voxel dropout `0.08`.",
        "",
        "| # | source | particle | hits | kept hits | kept energy | image |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in report_rows:
        image_rel = Path(row["image"]).relative_to(args.markdown.parent)
        lines.append(
            f"| {row['rank']} | `{row['source_path']}` | {row['particle_id']} | {row['n_hits']} | "
            f"{100.0 * float(row['kept_fraction']):.2f}% | {100.0 * float(row['kept_energy_fraction']):.2f}% | "
            f"[png]({image_rel.as_posix()}) |"
        )
    lines.append("")
    for row in report_rows:
        image_rel = Path(row["image"]).relative_to(args.markdown.parent)
        lines.extend([f"## Example {row['rank']}", "", f"![example {row['rank']}]({image_rel.as_posix()})", ""])
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(args.markdown)


if __name__ == "__main__":
    main()

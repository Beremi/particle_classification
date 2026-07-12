#!/usr/bin/env python3
"""Render held-out voxel autoencoder reconstruction examples."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from particle_classification.experiments.autoencoders.voxel import (
    VoxelGridConfig,
    VoxelMLPConfig,
    VoxelPatchMLPAutoencoder,
    VoxelSparseCache,
    corrupt_voxel_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blur-kernel", type=int, default=1)
    parser.add_argument("--blur-mix", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--voxel-dropout", type=float, default=0.0)
    parser.add_argument(
        "--include-energy-matched",
        action="store_true",
        help="Also plot reconstruction rescaled to the corrupted-input total energy for diagnostics.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> tuple[VoxelPatchMLPAutoencoder, VoxelGridConfig]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = VoxelMLPConfig(**checkpoint["model_config"])
    grid_config = VoxelGridConfig(**checkpoint["grid_config"])
    model = VoxelPatchMLPAutoencoder(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, grid_config


def four_time_slices(volume: np.ndarray) -> np.ndarray:
    chunks = np.array_split(volume, 4, axis=0)
    return np.stack([chunk.sum(axis=0) for chunk in chunks], axis=0)


def render_example(
    original: np.ndarray,
    model_input: np.ndarray,
    reconstruction: np.ndarray,
    row: dict[str, str],
    path: Path,
    *,
    include_energy_matched: bool = False,
) -> None:
    original_slices = four_time_slices(original)
    input_slices = four_time_slices(model_input)
    recon_slices = four_time_slices(reconstruction)
    energy_matched = None
    energy_matched_slices = None
    if include_energy_matched:
        input_energy = max(float(model_input.sum()), 1e-12)
        recon_energy = max(float(reconstruction.sum()), 1e-12)
        energy_matched = reconstruction * (input_energy / recon_energy)
        energy_matched_slices = four_time_slices(energy_matched)
    error_slices = np.abs(recon_slices - original_slices)
    vmax = max(
        float(original_slices.max()),
        float(input_slices.max()),
        float(recon_slices.max()),
        float(energy_matched_slices.max()) if energy_matched_slices is not None else 0.0,
        1e-6,
    )
    err_vmax = max(float(error_slices.max()), 1e-6)

    row_labels = ["original", "input", "recon"]
    image_rows = [original_slices, input_slices, recon_slices]
    if energy_matched_slices is not None:
        row_labels.append("recon E-matched")
        image_rows.append(energy_matched_slices)
    row_labels.append("|error|")
    image_rows.append(error_slices)

    fig, axes = plt.subplots(len(row_labels), 4, figsize=(11.5, 2.55 * len(row_labels)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    main_images = []
    error_images = []
    for col in range(4):
        axes[0, col].set_title(f"time block {col + 1}")
        for row_idx, images in enumerate(image_rows):
            is_error = row_labels[row_idx] == "|error|"
            image = axes[row_idx, col].imshow(
                images[col],
                origin="lower",
                cmap="viridis" if is_error else "magma",
                vmin=0,
                vmax=err_vmax if is_error else vmax,
            )
            (error_images if is_error else main_images).append(image)
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label)
    fig.colorbar(main_images[0], ax=axes[:-1, :], shrink=0.72, label="summed normalized log energy")
    fig.colorbar(error_images[0], ax=axes[-1, :], shrink=0.72, label="absolute error")
    fig.suptitle(
        f"{Path(row['source_path']).name} | particle {row['particle_id']} | hits {row['n_hits']} | "
        f"kept {float(row['kept_fraction']):.1%} | "
        f"E target/input/recon {original.sum():.2f}/{model_input.sum():.2f}/{reconstruction.sum():.2f}",
        fontsize=11,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def energy_row(idx: int, row: dict[str, str], original: np.ndarray, model_input: np.ndarray, reconstruction: np.ndarray) -> str:
    clean_energy = max(float(original.sum()), 1e-12)
    input_energy = float(model_input.sum())
    recon_energy = float(reconstruction.sum())
    return (
        f"| {idx} | {row['n_hits']} | {clean_energy:.3f} | {input_energy:.3f} | {recon_energy:.3f} | "
        f"{input_energy / clean_energy:.3f} | {recon_energy / clean_energy:.3f} | "
        f"{abs(recon_energy - clean_energy) / clean_energy:.3f} |"
    )


def main() -> None:
    args = parse_args()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    model, grid_config = load_model(args.checkpoint, args.device)
    device = next(model.parameters()).device
    dataset = VoxelSparseCache(args.cache, split=args.split, grid_config=grid_config, seed=args.seed)
    candidates = [row for row in dataset.rows if int(float(row.get("n_hits", 0))) >= args.min_hits]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.examples]
    if not selected:
        raise SystemExit("No matching rows found.")

    batch = dataset.rows_to_dense(selected, device=device)
    torch.manual_seed(args.seed)
    model_input = corrupt_voxel_batch(
        batch,
        blur_kernel=args.blur_kernel,
        blur_mix=args.blur_mix,
        noise_std=args.noise_std,
        voxel_dropout=args.voxel_dropout,
    )
    with torch.no_grad():
        reconstruction = model(model_input)["reconstruction"].detach().cpu().numpy()
    original = batch.detach().cpu().numpy()
    input_np = model_input.detach().cpu().numpy()

    lines = ["# Phase 2 Voxel Autoencoder Reconstruction Gallery", ""]
    lines.append(f"Checkpoint: `{args.checkpoint.as_posix()}`")
    lines.append("")
    lines.append("Each panel is a held-out test particle. The 32 time bins are summed into four 2D XY images.")
    lines.append("")
    lines.append(
        "Rows are clean target, corrupted model input, reconstruction from that corrupted input, and absolute reconstruction error."
    )
    lines.append("")
    lines.append(
        f"Input corruption: `blur_kernel={args.blur_kernel}`, `blur_mix={args.blur_mix}`, "
        f"`noise_std={args.noise_std}`, `voxel_dropout={args.voxel_dropout}`."
    )
    if args.include_energy_matched:
        lines.append("")
        lines.append(
            "The image panels include a diagnostic `recon E-matched` row: the raw reconstruction multiplied "
            "by `input_energy / reconstruction_energy`. This row is not the model output; it shows whether the "
            "remaining visual error is mostly global energy scale or geometry."
        )
    lines.append("")
    lines.append("| # | source | particle | hits | kept | image |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for idx, row in enumerate(selected, start=1):
        image_path = args.asset_dir / f"voxel_reconstruction_example_{idx:02d}.png"
        render_example(
            original[idx - 1],
            input_np[idx - 1],
            reconstruction[idx - 1],
            row,
            image_path,
            include_energy_matched=args.include_energy_matched,
        )
        rel_image = image_path.relative_to(args.out.parent).as_posix()
        lines.append(
            f"| {idx} | `{row['source_path']}` | {row['particle_id']} | {row['n_hits']} | "
            f"{float(row['kept_fraction']):.1%} | [png]({rel_image}) |"
        )
    lines.append("")
    lines.append("## Energy Check")
    lines.append("")
    lines.append("| # | hits | clean E | input E | recon E | input/clean | recon/clean | recon relative error |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(selected, start=1):
        lines.append(energy_row(idx, row, original[idx - 1], input_np[idx - 1], reconstruction[idx - 1]))
    lines.append("")
    for idx in range(1, len(selected) + 1):
        image_path = args.asset_dir / f"voxel_reconstruction_example_{idx:02d}.png"
        rel_image = image_path.relative_to(args.out.parent).as_posix()
        lines.append(f"## Example {idx}")
        lines.append("")
        lines.append(f"![example {idx}]({rel_image})")
        lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

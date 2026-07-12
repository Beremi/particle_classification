#!/usr/bin/env python3
"""Remake Shape-8 transform-pair examples as 3D cube plots."""

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from particle_classification.experiments.autoencoders.path import StructuredTransformAutoencoder, TransformAEConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("local_data/processed/phase2_structured_transform_cache_eps5_v001_probe"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "local_data/experiments/phase2_structured_transform_ae_eps5_shape8_outer_uniform/"
            "runs/structured_transform_ae_shape8_t7_p128_h1536/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("experimental_notes/assets/phase2_shape8_heavy_training/latent_transform_pairs/latent_pair_examples.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experimental_notes/assets/phase2_shape8_heavy_training/latent_transform_pairs"),
    )
    return parser.parse_args()


def cube_faces(center: np.ndarray, size: float) -> list[list[tuple[float, float, float]]]:
    x, y, z = center
    d = size * 0.5
    verts = [
        (x - d, y - d, z - d),
        (x + d, y - d, z - d),
        (x + d, y + d, z - d),
        (x - d, y + d, z - d),
        (x - d, y - d, z + d),
        (x + d, y - d, z + d),
        (x + d, y + d, z + d),
        (x - d, y + d, z + d),
    ]
    return [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[0], verts[3], verts[7], verts[4]],
    ]


def as_display_xyz(path: np.ndarray, xy_scale: float) -> np.ndarray:
    return np.column_stack([path[:, 0] * xy_scale, path[:, 1] * xy_scale, path[:, 2]])


def draw_cubes(
    ax: plt.Axes,
    path: np.ndarray,
    *,
    title: str,
    xy_scale: float,
    cube_size: float,
    cmap_name: str,
    norm: colors.Normalize,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    xyz = as_display_xyz(path, xy_scale)
    energy = np.asarray(path[:, 3], dtype=np.float32)
    cmap = plt.colormaps[cmap_name]
    faces: list[list[tuple[float, float, float]]] = []
    face_colors: list[tuple[float, float, float, float]] = []
    # Draw low-time cubes first. This gives the later/high-time cubes a clearer edge.
    for center, value in sorted(zip(xyz, energy, strict=True), key=lambda item: float(item[0][2])):
        rgba = cmap(norm(float(value)))
        rgba = (rgba[0], rgba[1], rgba[2], 0.48)
        cube = cube_faces(center.astype(float), cube_size)
        faces.extend(cube)
        face_colors.extend([rgba] * len(cube))
    collection = Poly3DCollection(
        faces,
        facecolors=face_colors,
        edgecolors=(0.05, 0.05, 0.05, 0.23),
        linewidths=0.16,
    )
    ax.add_collection3d(collection)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=6, c=energy, cmap=cmap, norm=norm, depthshade=False, alpha=0.85)
    ax.set_title(title, pad=8, fontsize=12)
    ax.set_xlabel(f"x * {xy_scale:.1f}")
    ax.set_ylabel(f"y * {xy_scale:.1f}")
    ax.set_zlabel("t centered")
    ax.view_init(elev=22, azim=-58)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.grid(True, alpha=0.25)


def equal_limits(paths: list[np.ndarray], xy_scale: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    xyz = np.concatenate([as_display_xyz(path, xy_scale) for path in paths], axis=0)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    span = float(np.max(maxs - mins))
    span = max(span, 0.08)
    pad = span * 0.10
    half = span * 0.5 + pad
    return tuple((float(c - half), float(c + half)) for c in center)  # type: ignore[return-value]


def xy_display_scale(paths: list[np.ndarray]) -> float:
    arr = np.concatenate(paths, axis=0)
    xy_span = max(float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1])), 1e-6)
    t_span = max(float(np.ptp(arr[:, 2])), 1e-6)
    return float(np.clip(t_span / xy_span, 1.0, 18.0))


def pair_cube_size(paths: list[np.ndarray], xy_scale: float) -> float:
    xyz = np.concatenate([as_display_xyz(path, xy_scale) for path in paths], axis=0)
    span = float(np.max(np.ptp(xyz, axis=0)))
    return max(0.018, span * 0.035)


def load_model(checkpoint: Path) -> StructuredTransformAutoencoder:
    payload = torch.load(checkpoint, map_location="cpu")
    model = StructuredTransformAutoencoder(TransformAEConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def transform_tail(row: dict[str, str], prefix: str) -> str:
    return (
        f"{prefix}: theta_xy={float(row[f'{prefix.lower()}_theta_xy']):+.2f}, "
        f"theta_time={float(row[f'{prefix.lower()}_theta_time']):+.2f}, "
        f"scale={float(row[f'{prefix.lower()}_scale_xyz']):.2f}, "
        f"energy_scale={float(row[f'{prefix.lower()}_energy_scale']):.2f}"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = np.load(args.cache / "test.npz")["path"].astype(np.float32)
    with args.pairs.open(newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))

    indices = sorted({int(row["i"]) for row in pair_rows} | {int(row["j"]) for row in pair_rows})
    index_to_pos = {idx: pos for pos, idx in enumerate(indices)}
    batch = torch.from_numpy(paths[indices])
    model = load_model(args.checkpoint)
    with torch.no_grad():
        recon = model(batch)["reconstruction"].cpu().numpy().astype(np.float32)

    for row in pair_rows:
        rank = int(row["rank"])
        i = int(row["i"])
        j = int(row["j"])
        target_a = paths[i]
        target_b = paths[j]
        recon_a = recon[index_to_pos[i]]
        recon_b = recon[index_to_pos[j]]
        shown = [target_a, recon_a, target_b, recon_b]
        xy_scale = xy_display_scale(shown)
        cube_size = pair_cube_size(shown, xy_scale)
        limits = equal_limits(shown, xy_scale)
        energy_values = np.concatenate([path[:, 3] for path in shown])
        norm = colors.Normalize(vmin=float(energy_values.min()), vmax=float(energy_values.max()))

        fig = plt.figure(figsize=(18, 14), dpi=170)
        grid = fig.add_gridspec(
            2,
            2,
            left=0.04,
            right=0.88,
            bottom=0.13,
            top=0.90,
            wspace=0.05,
            hspace=0.14,
        )
        fig.suptitle(
            f"Pair {rank}: close 8D shape latent, different explicit transform tail",
            fontsize=18,
            y=0.965,
        )
        panels = [
            (target_a, f"A target cubes | hits={row['a_hits']} | err={float(row['a_err']):.4f}", "viridis"),
            (recon_a, "A reconstruction cubes", "magma"),
            (target_b, f"B target cubes | hits={row['b_hits']} | err={float(row['b_err']):.4f}", "viridis"),
            (recon_b, "B reconstruction cubes", "magma"),
        ]
        axes = []
        for panel_idx, (path, title, cmap_name) in enumerate(panels):
            ax = fig.add_subplot(grid[panel_idx // 2, panel_idx % 2], projection="3d")
            axes.append(ax)
            draw_cubes(
                ax,
                path,
                title=title,
                xy_scale=xy_scale,
                cube_size=cube_size,
                cmap_name=cmap_name,
                norm=norm,
                limits=limits,
            )

        fig.text(
            0.5,
            0.055,
            (
                f"z_dist={float(row['z_dist']):.4f} | transform_dist={float(row['transform_dist']):.4f} | "
                f"canonical_RMSE={float(row['canonical_rmse']):.4f} | recon_RMSE={float(row['reconstruction_rmse']):.4f}\n"
                f"{transform_tail(row, 'A')}    {transform_tail(row, 'B')}"
            ),
            ha="center",
            va="center",
            fontsize=11,
        )
        scalar = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        scalar.set_array([])
        cbar_ax = fig.add_axes([0.91, 0.27, 0.018, 0.46])
        cbar = fig.colorbar(scalar, cax=cbar_ax)
        cbar.set_label("normalized log ToT energy")
        output = args.out_dir / f"latent_pair_{rank:02d}.png"
        fig.savefig(output)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()

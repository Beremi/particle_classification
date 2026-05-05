#!/usr/bin/env python3
"""Open a focused 3D Matplotlib view of one particle plus nearby noise hits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree


FINE_TIME_BIN_NS = 25.0 / 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--z-start", type=int, required=True, help="Start fine-time bin, i.e. round(hit_time * 16).")
    parser.add_argument("--z-end", type=int, required=True, help="End fine-time bin, i.e. round(hit_time * 16).")
    parser.add_argument("--particle-id", type=int, required=True)
    parser.add_argument("--time-scale", type=float, default=0.625)
    parser.add_argument("--eps", type=float, default=5.0)
    parser.add_argument("--noise-count", type=int, default=2)
    parser.add_argument("--noise-max-distance", type=float, default=20.0)
    parser.add_argument("--backend", default="gtk4agg")
    parser.add_argument("--point-size", type=float, default=34.0)
    parser.add_argument("--noise-size", type=float, default=105.0)
    parser.add_argument("--alpha", type=float, default=0.92)
    return parser.parse_args()


def load_window(args: argparse.Namespace) -> dict[str, np.ndarray | str]:
    with np.load(args.npz, allow_pickle=False) as data:
        hit_time = data["hit_time"].astype(np.float64)
        z_fine = np.rint(hit_time * 16.0).astype(np.int64)
        mask = (z_fine >= args.z_start) & (z_fine < args.z_end)
        if not np.any(mask):
            raise SystemExit("Selected window has no hits.")
        return {
            "x": data["hit_x"][mask].astype(np.float64),
            "y": data["hit_y"][mask].astype(np.float64),
            "z": (hit_time[mask] - (args.z_start / 16.0)) / args.time_scale,
            "energy": data["hit_energy"][mask].astype(np.float64),
            "labels": data["hit_particle_id"][mask].astype(np.int32),
            "source": str(data["source_path"]),
        }


def equalish_axes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    pad = 3.0
    xmin, xmax = float(np.min(x) - pad), float(np.max(x) + pad)
    ymin, ymax = float(np.min(y) - pad), float(np.max(y) + pad)
    zmin, zmax = float(np.min(z) - pad), float(np.max(z) + pad)
    span = max(xmax - xmin, ymax - ymin, (zmax - zmin) / 0.72)
    xc, yc, zc = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    ax.set_xlim(xc - span / 2, xc + span / 2)
    ax.set_ylim(yc - span / 2, yc + span / 2)
    ax.set_zlim(zc - span * 0.72 / 2, zc + span * 0.72 / 2)
    ax.set_box_aspect((1.0, 1.0, 0.72))


def main() -> None:
    args = parse_args()
    matplotlib.use(args.backend, force=True)
    import matplotlib.pyplot as plt

    data = load_window(args)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    energy = data["energy"]
    labels = data["labels"]
    source = str(data["source"])

    particle_mask = labels == args.particle_id
    if not np.any(particle_mask):
        raise SystemExit(f"Particle ID {args.particle_id} is not present in the selected window.")

    noise_mask = labels < 0
    px, py, pz = x[particle_mask], y[particle_mask], z[particle_mask]
    particle_points = np.column_stack([px, py, pz])
    selected_noise = np.zeros(len(x), dtype=bool)
    nearest_lines: list[tuple[np.ndarray, np.ndarray, float]] = []

    if np.any(noise_mask):
        noise_indices = np.flatnonzero(noise_mask)
        tree = cKDTree(particle_points)
        noise_points = np.column_stack([x[noise_mask], y[noise_mask], z[noise_mask]])
        distances, nearest = tree.query(noise_points, k=1)
        order = np.argsort(distances)
        kept = []
        for noise_pos in order:
            if len(kept) >= args.noise_count:
                break
            if distances[noise_pos] <= args.noise_max_distance:
                kept.append(noise_pos)
        selected_indices = noise_indices[np.array(kept, dtype=int)] if kept else np.array([], dtype=int)
        selected_noise[selected_indices] = True
        for noise_pos, hit_index in zip(kept, selected_indices, strict=True):
            start = np.array([x[hit_index], y[hit_index], z[hit_index]])
            end = particle_points[nearest[noise_pos]]
            nearest_lines.append((start, end, float(distances[noise_pos])))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        px,
        py,
        pz,
        c=energy[particle_mask],
        cmap="plasma",
        s=args.point_size,
        alpha=args.alpha,
        linewidths=0,
        label=f"particle {args.particle_id}",
    )
    if np.any(selected_noise):
        ax.scatter(
            x[selected_noise],
            y[selected_noise],
            z[selected_noise],
            c="red",
            marker="X",
            s=args.noise_size,
            alpha=1.0,
            linewidths=1.0,
            edgecolors="black",
            label="nearby DBSCAN noise",
        )
    for start, end, distance in nearest_lines:
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="red", linewidth=1.8)
        mid = (start + end) / 2
        ax.text(mid[0], mid[1], mid[2], f"{distance:.2f}", color="red", fontsize=9)

    fig.colorbar(sc, ax=ax, shrink=0.72, label="log1p(ToT)")
    all_x = np.concatenate([px, x[selected_noise]]) if np.any(selected_noise) else px
    all_y = np.concatenate([py, y[selected_noise]]) if np.any(selected_noise) else py
    all_z = np.concatenate([pz, z[selected_noise]]) if np.any(selected_noise) else pz
    equalish_axes(ax, all_x, all_y, all_z)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel(f"scaled time / {args.time_scale:g}")
    ax.view_init(elev=23, azim=-58)
    span_us = (args.z_end - args.z_start) * FINE_TIME_BIN_NS / 1000.0
    ax.set_title(f"{Path(source).name} | particle {args.particle_id} + nearby noise | eps={args.eps:g}")
    fig.suptitle(
        f"window {args.z_start}..{args.z_end} fine bins ({span_us:.1f} us); "
        f"red labels are nearest DBSCAN-noise distances to this particle",
        fontsize=10,
    )
    ax.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()

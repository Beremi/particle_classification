#!/usr/bin/env python3
"""Open an interactive Matplotlib 3D view using DBSCAN-scaled time."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np


FINE_TIME_BIN_NS = 25.0 / 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--z-start", type=int, required=True)
    parser.add_argument("--z-end", type=int, required=True)
    parser.add_argument("--time-scale", type=float, default=0.625)
    parser.add_argument("--backend", default="gtk4agg")
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--alpha", type=float, default=0.86)
    parser.add_argument("--single", choices=["labels", "energy"], default=None)
    return parser.parse_args()


def load_window(args: argparse.Namespace) -> dict[str, np.ndarray | str]:
    with np.load(args.npz, allow_pickle=False) as data:
        z_fine = np.rint(data["hit_time"].astype(np.float64) * 16.0).astype(np.int64)
        mask = (z_fine >= args.z_start) & (z_fine < args.z_end)
        if not np.any(mask):
            raise SystemExit("Selected window has no hits.")
        return {
            "x": data["hit_x"][mask].astype(np.float64),
            "y": data["hit_y"][mask].astype(np.float64),
            "z": (data["hit_time"][mask].astype(np.float64) - (args.z_start / 16.0)) / args.time_scale,
            "energy": data["hit_energy"][mask].astype(np.float64),
            "labels": data["hit_particle_id"][mask].astype(np.int32),
            "source": str(data["source_path"]),
        }


def label_values(labels: np.ndarray) -> np.ndarray:
    values = np.zeros(len(labels), dtype=np.float64)
    valid = labels >= 0
    if np.any(valid):
        values[valid] = ((labels[valid].astype(np.uint64) * np.uint64(2654435761)) % np.uint64(4096)).astype(float)
    values[~valid] = np.nan
    return values


def configure_axis(ax, *, z_span: float, time_scale: float) -> None:
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_zlim(0, z_span)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel(f"scaled time / {time_scale:g}")
    ax.set_box_aspect((1.0, 1.0, 0.72))
    ax.view_init(elev=23, azim=-58)


def main() -> None:
    args = parse_args()
    matplotlib.use(args.backend, force=True)
    import matplotlib.pyplot as plt

    data = load_window(args)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    labels = data["labels"]
    energy = data["energy"]
    source = str(data["source"])
    z_span = (args.z_end - args.z_start) / (16.0 * args.time_scale)
    n_particles = len(np.unique(labels[labels >= 0]))
    noise_fraction = float(np.mean(labels < 0))
    span_us = (args.z_end - args.z_start) * FINE_TIME_BIN_NS / 1000.0

    if args.single:
        fig = plt.figure(figsize=(11, 8))
        axes = [fig.add_subplot(111, projection="3d")]
        modes = [args.single]
    else:
        fig = plt.figure(figsize=(15, 7))
        axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]
        modes = ["labels", "energy"]

    for ax, mode in zip(axes, modes, strict=True):
        if mode == "labels":
            valid = labels >= 0
            ax.scatter(x[~valid], y[~valid], z[~valid], c="lightgray", s=args.point_size, alpha=0.24, linewidths=0)
            ax.scatter(
                x[valid],
                y[valid],
                z[valid],
                c=label_values(labels)[valid],
                cmap="tab20",
                s=args.point_size,
                alpha=args.alpha,
                linewidths=0,
            )
            ax.set_title("DBSCAN particle IDs")
        else:
            sc = ax.scatter(x, y, z, c=energy, cmap="plasma", s=args.point_size, alpha=args.alpha, linewidths=0)
            fig.colorbar(sc, ax=ax, shrink=0.70, label="log1p(ToT)")
            ax.set_title("energy")
        configure_axis(ax, z_span=z_span, time_scale=args.time_scale)

    fig.suptitle(
        f"{Path(source).name} | {len(x)} hits | {n_particles} particles | "
        f"noise {noise_fraction:.2%} | {span_us:.1f} us | DBSCAN-scaled time",
        fontsize=11,
    )
    plt.show()


if __name__ == "__main__":
    main()

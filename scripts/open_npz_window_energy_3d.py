#!/usr/bin/env python3
"""Open an interactive Matplotlib 3D energy view for an NPZ time window."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


FINE_TIME_BIN_NS = 25.0 / 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--z-start", type=int, required=True, help="Absolute fine-time-bin start.")
    parser.add_argument("--z-end", type=int, required=True, help="Absolute fine-time-bin end.")
    parser.add_argument("--plot-time-bin-factor", type=int, default=100)
    parser.add_argument("--backend", default="QtAgg")
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--mode", choices=["scatter", "cubes"], default="scatter")
    return parser.parse_args()


def load_window(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(args.npz, allow_pickle=False) as data:
        z = np.rint(data["hit_time"].astype(np.float64) * 16.0).astype(np.int64)
        mask = (z >= args.z_start) & (z < args.z_end)
        x = data["hit_x"][mask].astype(np.float64)
        y = data["hit_y"][mask].astype(np.float64)
        z_local = ((z[mask] - args.z_start) // args.plot_time_bin_factor).astype(np.float64)
        energy = data["hit_energy"][mask].astype(np.float64)
        source = str(data["source_path"])
    if len(x) == 0:
        raise SystemExit("Selected window has no hits.")
    return x, y, z_local, energy, source


def energy_colors(energy: np.ndarray, alpha: float) -> np.ndarray:
    norm = Normalize(
        vmin=float(np.quantile(energy, 0.05)),
        vmax=float(np.quantile(energy, 0.95)),
        clip=True,
    )
    colors = cm.plasma(norm(energy))
    colors[:, 3] = alpha
    return colors


def draw_cubes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray) -> None:
    faces: list[list[tuple[float, float, float]]] = []
    facecolors: list[tuple[float, float, float, float]] = []
    for x0, y0, z0, color in zip(x, y, z, colors, strict=False):
        x1, y1, z1 = x0 + 1.0, y0 + 1.0, z0 + 1.0
        cube_faces = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
        faces.extend(cube_faces)
        facecolors.extend([tuple(float(v) for v in color)] * 6)
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=facecolors,
            edgecolors=(0.0, 0.0, 0.0, 0.12),
            linewidths=0.12,
        )
    )


def main() -> None:
    args = parse_args()
    matplotlib.use(args.backend, force=True)
    import matplotlib.pyplot as plt

    x, y, z, energy, source = load_window(args)
    colors = energy_colors(energy, args.alpha)
    z_span = int(np.ceil((args.z_end - args.z_start) / args.plot_time_bin_factor))
    span_ns = (args.z_end - args.z_start) * FINE_TIME_BIN_NS
    z_cube_ns = args.plot_time_bin_factor * FINE_TIME_BIN_NS

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    if args.mode == "cubes":
        draw_cubes(ax, x, y, z, colors)
    else:
        sc = ax.scatter(x + 0.5, y + 0.5, z + 0.5, c=energy, cmap="plasma", s=args.point_size, alpha=args.alpha)
        fig.colorbar(sc, ax=ax, shrink=0.72, label="log1p(ToT)")

    ax.set_title(
        f"{Path(source).name} | energy | {len(x)} hits | "
        f"{span_ns:.0f} ns window | z cube {z_cube_ns:.2f} ns"
    )
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel("time bin")
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_zlim(0, z_span)
    ax.set_box_aspect((256, 256, z_span))
    ax.view_init(elev=24, azim=-58)
    plt.show()


if __name__ == "__main__":
    main()

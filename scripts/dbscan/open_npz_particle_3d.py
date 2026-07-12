#!/usr/bin/env python3
"""Open an interactive 3D Matplotlib view of one extracted particle."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np


FINE_TIME_BIN_NS = 25.0 / 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="Particle NPZ shard.")
    parser.add_argument("--particle-id", type=int, required=True, help="Particle ID from hit_particle_id/particle_id.")
    parser.add_argument("--time-scale", type=float, default=0.625, help="Time scale used on the z axis.")
    parser.add_argument(
        "--component-time-bin",
        type=float,
        default=0.625,
        help="Time bin for 26-neighbor cube connectivity diagnostics.",
    )
    parser.add_argument("--backend", default="gtk4agg", help="Matplotlib GUI backend, e.g. gtk4agg or tkagg.")
    parser.add_argument("--point-size", type=float, default=46.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--mode", choices=["energy", "components", "both"], default="both")
    parser.add_argument("--show-singletons", action="store_true", help="Color singleton components instead of gray.")
    return parser.parse_args()


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def load_particle(npz_path: Path, particle_id: int) -> dict[str, np.ndarray | str | int]:
    with np.load(npz_path, allow_pickle=False) as data:
        particle_ids = data["particle_id"].astype(np.int64)
        matches = np.flatnonzero(particle_ids == particle_id)
        if len(matches) == 0:
            raise SystemExit(f"Particle ID {particle_id} was not found in {npz_path}.")
        particle_index = int(matches[0])
        start = int(data["particle_offsets"][particle_index])
        end = int(data["particle_offsets"][particle_index + 1])
        return {
            "particle_index": particle_index,
            "source": str(data["source_path"]),
            "x": data["hit_x"][start:end].astype(np.float64),
            "y": data["hit_y"][start:end].astype(np.float64),
            "time": data["hit_time"][start:end].astype(np.float64),
            "energy": data["hit_energy"][start:end].astype(np.float64),
        }


def connected_components_26(x: np.ndarray, y: np.ndarray, time: np.ndarray, time_bin: float) -> tuple[np.ndarray, list[int]]:
    if len(x) == 0:
        return np.empty(0, dtype=np.int32), []
    z = np.rint((time - float(np.min(time))) / time_bin).astype(np.int64)
    coords = np.column_stack([x.astype(np.int64), y.astype(np.int64), z])
    uf = UnionFind(len(coords))
    first_at_coord: dict[tuple[int, int, int], int] = {}

    for i, coord_array in enumerate(coords):
        coord = tuple(int(v) for v in coord_array)
        previous = first_at_coord.get(coord)
        if previous is None:
            first_at_coord[coord] = i
        else:
            uf.union(i, previous)

        cx, cy, cz = coord
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    j = first_at_coord.get((cx + dx, cy + dy, cz + dz))
                    if j is not None:
                        uf.union(i, j)

    roots = np.array([uf.find(i) for i in range(len(coords))], dtype=np.int32)
    counts = Counter(int(root) for root in roots)
    roots_by_size = [root for root, _ in counts.most_common()]
    root_to_rank = {root: rank for rank, root in enumerate(roots_by_size)}
    component_rank = np.array([root_to_rank[int(root)] for root in roots], dtype=np.int32)
    sizes = [counts[root] for root in roots_by_size]
    return component_rank, sizes


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


def configure_axis(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, time_scale: float) -> None:
    equalish_axes(ax, x, y, z)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel(f"relative time / {time_scale:g}")
    ax.view_init(elev=23, azim=-58)


def main() -> None:
    args = parse_args()
    matplotlib.use(args.backend, force=True)
    import matplotlib.pyplot as plt

    particle = load_particle(args.npz, args.particle_id)
    x = particle["x"]
    y = particle["y"]
    time = particle["time"]
    energy = particle["energy"]
    z = (time - float(np.min(time))) / args.time_scale
    component_rank, component_sizes = connected_components_26(x, y, time, args.component_time_bin)

    print(f"source: {particle['source']}")
    print(f"particle_id: {args.particle_id}; particle_index: {particle['particle_index']}; hits: {len(x)}")
    print(f"x: {np.min(x):.0f}..{np.max(x):.0f}; y: {np.min(y):.0f}..{np.max(y):.0f}")
    print(f"time: {np.min(time):.6f}..{np.max(time):.6f}; span: {np.max(time) - np.min(time):.6f} ticks")
    print(f"energy log1p(ToT): {np.min(energy):.4f}..{np.max(energy):.4f}")
    print(f"26-neighbor components at time_bin={args.component_time_bin:g}: {len(component_sizes)}")
    print("largest component sizes:", component_sizes[:20])

    modes = ["energy", "components"] if args.mode == "both" else [args.mode]
    fig = plt.figure(figsize=(15, 7) if len(modes) == 2 else (10, 8))
    axes = [fig.add_subplot(1, len(modes), i + 1, projection="3d") for i in range(len(modes))]

    for ax, mode in zip(axes, modes, strict=True):
        if mode == "energy":
            sc = ax.scatter(x, y, z, c=energy, cmap="plasma", s=args.point_size, alpha=args.alpha, linewidths=0)
            fig.colorbar(sc, ax=ax, shrink=0.72, label="log1p(ToT)")
            ax.set_title("original hits colored by energy")
        else:
            singleton = np.array([component_sizes[int(rank)] == 1 for rank in component_rank], dtype=bool)
            if np.any(singleton) and not args.show_singletons:
                ax.scatter(x[singleton], y[singleton], z[singleton], c="lightgray", s=args.point_size, alpha=0.45)
                main_mask = ~singleton
            else:
                main_mask = np.ones(len(x), dtype=bool)
            sc = ax.scatter(
                x[main_mask],
                y[main_mask],
                z[main_mask],
                c=component_rank[main_mask],
                cmap="tab20",
                s=args.point_size,
                alpha=args.alpha,
                linewidths=0,
            )
            fig.colorbar(sc, ax=ax, shrink=0.72, label="component rank by size")
            ax.set_title(f"26-neighbor components, time bin {args.component_time_bin:g}")
        configure_axis(ax, x, y, z, args.time_scale)

    span_us = (np.max(time) - np.min(time)) * 25.0 / 1000.0
    size_preview = ", ".join(str(v) for v in component_sizes[:8])
    fig.suptitle(
        f"{Path(str(particle['source'])).name} | particle {args.particle_id} | {len(x)} hits | "
        f"{span_us:.3f} us span | largest components: {size_preview}",
        fontsize=11,
    )
    plt.show()


if __name__ == "__main__":
    main()

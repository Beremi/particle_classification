#!/usr/bin/env python3
"""Render native-DBSCAN largest particles as real voxel cubes.

The z axis is expressed in fine Timepix time bins:

    z_bin = round((ToA - FToA / 16) * 16)

so one z unit is one FToA sub-tick, i.e. 25 ns / 16 = 1.5625 ns.
The matplotlib box aspect is set from the data ranges so a unit in x, y,
and z is drawn with the same scale.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize


FINE_TIME_BIN_NS = 25.0 / 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--largest-csv", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--alpha", type=float, default=0.38)
    parser.add_argument("--title", default="Native Grid DBSCAN Fine-Time Voxel-Cube Gallery")
    parser.add_argument(
        "--summary",
        default=(
            "Transparent cube rendering of the largest labels from the fine-time "
            "native-grid DBSCAN sample. One cube is one detector pixel by one "
            "fine time bin."
        ),
    )
    return parser.parse_args()


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        if parent != item:
            self.parent[item] = self.find(parent)
        return int(self.parent[item])

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def component_labels(coords: np.ndarray, *, mode: str) -> tuple[np.ndarray, int]:
    if len(coords) == 0:
        return np.empty(0, dtype=np.int32), 0
    if mode == "face":
        offsets = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
    elif mode == "corner":
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
    else:
        raise ValueError(f"Unknown component mode: {mode}")

    lookup = {tuple(map(int, coord)): idx for idx, coord in enumerate(coords)}
    labels = np.full(len(coords), -1, dtype=np.int32)
    next_label = 0
    for seed in range(len(coords)):
        if labels[seed] >= 0:
            continue
        labels[seed] = next_label
        queue: deque[int] = deque([seed])
        while queue:
            idx = queue.popleft()
            x, y, z = map(int, coords[idx])
            for dx, dy, dz in offsets:
                other = lookup.get((x + dx, y + dy, z + dz))
                if other is not None and labels[other] < 0:
                    labels[other] = next_label
                    queue.append(other)
        next_label += 1
    return labels, next_label


def particle_voxels(npz_path: Path, start: int, end: int) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        x = data["hit_x"][start:end].astype(np.int32)
        y = data["hit_y"][start:end].astype(np.int32)
        time = data["hit_time"][start:end].astype(np.float64)
        energy = data["hit_energy"][start:end].astype(np.float64)

    z = np.rint(time * 16.0).astype(np.int64)
    unique, inverse = np.unique(np.column_stack([x, y, z]), axis=0, return_inverse=True)
    energy_sum = np.bincount(inverse, weights=energy, minlength=len(unique))
    hit_count = np.bincount(inverse, minlength=len(unique))
    energy_mean = energy_sum / np.maximum(hit_count, 1)
    return {
        "coords": unique.astype(np.int64, copy=False),
        "energy_mean": energy_mean,
        "hit_count": hit_count,
        "raw_x": x,
        "raw_y": y,
        "raw_z": z,
    }


def build_dense_voxels(
    coords: np.ndarray,
    colors: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int, int, int]]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    shape = tuple((maxs - mins + 1).astype(int))
    filled = np.zeros(shape, dtype=bool)
    facecolors = np.zeros(shape + (4,), dtype=np.float32)
    rel = (coords - mins).astype(int)
    filled[rel[:, 0], rel[:, 1], rel[:, 2]] = True
    rgba = colors.copy()
    rgba[:, 3] = alpha
    facecolors[rel[:, 0], rel[:, 1], rel[:, 2]] = rgba

    x_edges = np.arange(int(mins[0]), int(maxs[0]) + 2)
    y_edges = np.arange(int(mins[1]), int(maxs[1]) + 2)
    z_edges = np.arange(0, int(maxs[2] - mins[2]) + 2)
    x_grid, y_grid, z_grid = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    bounds = (int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), int(mins[2]), int(maxs[2]))
    return filled, facecolors, x_grid, y_grid, z_grid, bounds


def set_real_cube_aspect(ax: plt.Axes, bounds: tuple[int, int, int, int, int, int]) -> None:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x_span = max(1, xmax - xmin + 1)
    y_span = max(1, ymax - ymin + 1)
    z_span = max(1, zmax - zmin + 1)
    ax.set_box_aspect((x_span, y_span, z_span))
    ax.set_xlim(xmin, xmax + 1)
    ax.set_ylim(ymin, ymax + 1)
    ax.set_zlim(0, z_span)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel("fine time bin")
    ax.view_init(elev=24, azim=-58)


def render_particle(row: pd.Series, out_path: Path, *, alpha: float) -> dict[str, int | float]:
    vox = particle_voxels(Path(row["shard"]), int(row["start"]), int(row["end"]))
    coords = vox["coords"]
    energy = vox["energy_mean"]
    face_labels, face_count = component_labels(coords, mode="face")
    corner_labels, corner_count = component_labels(coords, mode="corner")

    energy_norm = Normalize(
        vmin=float(np.quantile(energy, 0.05)) if len(energy) else 0.0,
        vmax=float(np.quantile(energy, 0.95)) if len(energy) else 1.0,
        clip=True,
    )
    energy_colors = cm.plasma(energy_norm(energy))

    if face_count <= 1:
        comp_values = np.zeros(len(face_labels), dtype=float)
    else:
        comp_values = face_labels.astype(float) / max(1, face_count - 1)
    comp_colors = cm.tab20(comp_values % 1.0)

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]

    for ax, colors, title in [
        (axes[0], energy_colors, "energy-colored DBSCAN label"),
        (axes[1], comp_colors, "strict face components"),
    ]:
        filled, facecolors, x_edges, y_edges, z_edges, bounds = build_dense_voxels(coords, colors, alpha=alpha)
        ax.voxels(
            x_edges,
            y_edges,
            z_edges,
            filled,
            facecolors=facecolors,
            edgecolors=(0.0, 0.0, 0.0, 0.16),
            linewidth=0.18,
        )
        set_real_cube_aspect(ax, bounds)
        ax.set_title(title, pad=8)

    z_span_bins = int(coords[:, 2].max() - coords[:, 2].min() + 1)
    fig.suptitle(
        f"rank {int(row['rank'])} | {row['source_path']} | particle {int(row['particle_id'])} | "
        f"{int(row['n_hits'])} hits | {z_span_bins} fine bins = {z_span_bins * FINE_TIME_BIN_NS:.2f} ns",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "occupied_voxels": int(len(coords)),
        "face_components_fine": int(face_count),
        "corner_components_fine": int(corner_count),
        "z_span_bins": int(z_span_bins),
        "z_span_ns": float(z_span_bins * FINE_TIME_BIN_NS),
    }


def source_table(manifest_path: Path | None) -> str:
    if manifest_path is None or not manifest_path.exists():
        return ""
    df = pd.read_csv(manifest_path)
    if df.empty:
        return ""
    lines = [
        "| source | hits | particles | noise frac | runtime s | hits/s | warnings |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| `{row.get('source_path', '')}` | {int(row.get('row_count', 0)):,} | "
            f"{int(row.get('particle_count', 0)):,} | {float(row.get('noise_fraction', 0.0)):.3f} | "
            f"{float(row.get('runtime_s', row.get('total_runtime_s', 0.0))):.3f} | "
            f"{float(row.get('hits_per_s', 0.0)):,.0f} | `{row.get('validation_warnings', '')}` |"
        )
    return "\n".join(lines)


def write_report(args: argparse.Namespace, rows: list[dict[str, object]], source_manifest: Path | None) -> None:
    rel_assets = args.assets.relative_to(args.report.parent)
    lines = [
        f"# {args.title}",
        "",
        args.summary,
        "",
        "The z axis is `round((ToA - FToA / 16) * 16)`, so one z-unit is one fine Timepix sub-tick "
        f"({FINE_TIME_BIN_NS:.4f} ns). Axes use equal data scaling: one x pixel, one y pixel, and one fine-time "
        "bin are rendered as the same cube size.",
        "",
    ]
    table = source_table(source_manifest)
    if table:
        lines += ["## Input Files", "", table, ""]
    hist = args.assets / "particle_size_histograms.png"
    if hist.exists():
        lines += [f"![particle size histograms]({rel_assets / hist.name})", ""]

    lines += [
        "## Largest Labels As Cubes",
        "",
        "Each image has two panels: left is the full DBSCAN label colored by energy; right recolors the same occupied "
        "cubes by strict face-connected components. If the right panel has many colors, the label is still only "
        "corner/edge-connected rather than face-continuous.",
        "",
        "| rank | source | particle | hits | voxels | z span ns | face comps | corner comps | image |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        img = Path(str(row["image"]))
        lines.append(
            f"| {int(row['rank'])} | `{row['source_path']}` | {int(row['particle_id'])} | "
            f"{int(row['n_hits']):,} | {int(row['occupied_voxels']):,} | {float(row['z_span_ns']):.2f} | "
            f"{int(row['face_components_fine'])} | {int(row['corner_components_fine'])} | "
            f"[png]({img.relative_to(args.report.parent)}) |"
        )

    for row in rows:
        img = Path(str(row["image"]))
        lines += [
            "",
            f"### Rank {int(row['rank'])}",
            "",
            f"![rank {int(row['rank'])}]({img.relative_to(args.report.parent)})",
        ]

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    largest = pd.read_csv(args.largest_csv).head(args.limit)
    rendered: list[dict[str, object]] = []
    for _, row in largest.iterrows():
        source_stem = Path(str(row["source_path"])).stem
        image = args.assets / f"largest_{int(row['rank']):02d}_{source_stem}_p{int(row['particle_id'])}_cubes.png"
        metrics = render_particle(row, image, alpha=args.alpha)
        record = row.to_dict()
        record.update(metrics)
        record["image"] = image.as_posix()
        rendered.append(record)
        print(f"rendered {image}")

    pd.DataFrame(rendered).to_csv(args.assets / "largest_particles_cube_gallery.csv", index=False)
    manifest = None
    if rendered:
        first_shard = Path(str(rendered[0]["shard"]))
        maybe_manifest = first_shard.parents[1] / "manifest.csv"
        if maybe_manifest.exists():
            manifest = maybe_manifest
    write_report(args, rendered, manifest)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

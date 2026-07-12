#!/usr/bin/env python3
"""Render whole-file native-DBSCAN inspection views as voxel-cube time windows.

Rendering every hit from a multi-million-hit file as literal equal-scale cubes is
not visually useful: the acquisition time axis can span billions of fine bins.
This script therefore writes whole-file context plots plus an equal-cube atlas of
the densest short time windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


FINE_TIME_BIN_NS = 25.0 / 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="Particle NPZ shard to inspect.")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument(
        "--window-start-fine-bin",
        action="append",
        type=int,
        default=[],
        help="Explicit absolute fine-time window start. Can be passed multiple times.",
    )
    parser.add_argument("--window-fine-bins", type=int, default=32, help="Time-window width in FToA sub-tick bins.")
    parser.add_argument(
        "--plot-time-bin-factor",
        type=int,
        default=1,
        help=(
            "Number of fine time bins collapsed into one plotted cube along z. "
            "Use 10 to show a 10x wider physical time slice with similar visual depth."
        ),
    )
    parser.add_argument("--min-window-gap", type=int, default=4, help="Minimum gap between selected window buckets.")
    parser.add_argument(
        "--select-by",
        choices=["hits", "particles"],
        default="hits",
        help="Rank time windows by hit count or by number of non-noise DBSCAN particles.",
    )
    parser.add_argument("--alpha", type=float, default=0.32)
    parser.add_argument(
        "--fixed-detector-xy",
        action="store_true",
        help="Show the full 0..256 detector plane instead of zooming to occupied x/y bounds.",
    )
    parser.add_argument(
        "--fixed-window-z",
        action="store_true",
        help="Show the full selected time slab on z instead of zooming to occupied z bins.",
    )
    parser.add_argument("--title", default="Whole-File Fine-Time Native DBSCAN Cube Overview")
    return parser.parse_args()


def load_hits(npz_path: Path) -> dict[str, np.ndarray | str | float]:
    with np.load(npz_path, allow_pickle=False) as data:
        source_path = str(data["source_path"])
        tick_ns = float(data["time_tick_ns"]) if "time_tick_ns" in data.files else 25.0
        return {
            "x": data["hit_x"].astype(np.int32),
            "y": data["hit_y"].astype(np.int32),
            "z": np.rint(data["hit_time"].astype(np.float64) * 16.0).astype(np.int64),
            "energy": data["hit_energy"].astype(np.float64),
            "labels": data["hit_particle_id"].astype(np.int32),
            "source_path": source_path,
            "tick_ns": tick_ns,
        }


def select_dense_windows(
    z: np.ndarray,
    labels: np.ndarray,
    *,
    window_bins: int,
    count: int,
    min_gap: int,
    select_by: str,
) -> list[tuple[int, int]]:
    bucket = z // window_bins
    sort_order = np.argsort(bucket, kind="stable")
    sorted_bucket = bucket[sort_order]
    sorted_labels = labels[sort_order]
    unique, starts, counts = np.unique(sorted_bucket, return_index=True, return_counts=True)
    if select_by == "hits":
        scores = counts
    elif select_by == "particles":
        scores = np.zeros(len(unique), dtype=np.int64)
        for i, (start, n_items) in enumerate(zip(starts, counts, strict=False)):
            chunk = sorted_labels[start : start + n_items]
            scores[i] = len(np.unique(chunk[chunk >= 0]))
    else:
        raise ValueError(f"Unknown window selection mode: {select_by}")
    order = np.lexsort((counts, scores))[::-1]
    selected: list[tuple[int, int]] = []
    selected_buckets: list[int] = []
    for idx in order:
        b = int(unique[idx])
        if any(abs(b - other) < min_gap for other in selected_buckets):
            continue
        selected.append((b * window_bins, int(scores[idx])))
        selected_buckets.append(b)
        if len(selected) >= count:
            break
    return selected


def voxelize_window(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,
    labels: np.ndarray,
    *,
    z_start: int,
    z_end: int,
    plot_time_bin_factor: int,
) -> dict[str, np.ndarray]:
    if plot_time_bin_factor < 1:
        raise ValueError("plot_time_bin_factor must be >= 1")
    mask = (z >= z_start) & (z < z_end)
    wx = x[mask]
    wy = y[mask]
    wz = (z[mask] - z_start) // plot_time_bin_factor
    wenergy = energy[mask]
    wlabels = labels[mask]
    coords = np.column_stack([wx, wy, wz]).astype(np.int64, copy=False)
    unique, first, inverse = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    energy_sum = np.bincount(inverse, weights=wenergy, minlength=len(unique))
    hit_count = np.bincount(inverse, minlength=len(unique))
    energy_mean = energy_sum / np.maximum(hit_count, 1)
    label_first = wlabels[first]
    # If several hits collapse into one displayed time cube, color the cube by the
    # majority non-noise label so the DBSCAN panel stays interpretable.
    if plot_time_bin_factor > 1 and len(unique):
        label_majority = np.full(len(unique), -1, dtype=np.int32)
        for voxel_idx in range(len(unique)):
            chunk = wlabels[inverse == voxel_idx]
            nonnoise = chunk[chunk >= 0]
            if len(nonnoise):
                values, counts = np.unique(nonnoise, return_counts=True)
                label_majority[voxel_idx] = int(values[np.argmax(counts)])
        label_first = label_majority
    return {
        "coords": unique,
        "energy_mean": energy_mean,
        "labels": label_first,
        "hit_count": hit_count,
        "raw_labels": wlabels,
        "raw_hit_count": np.asarray([len(wx)], dtype=np.int64),
    }


def stable_label_colors(labels: np.ndarray, *, alpha: float) -> np.ndarray:
    colors = np.zeros((len(labels), 4), dtype=np.float32)
    nonnoise = labels >= 0
    if np.any(nonnoise):
        # Multiplicative hash so neighboring particle IDs do not all get nearby colors.
        hashed = ((labels[nonnoise].astype(np.uint64) * np.uint64(2654435761)) % np.uint64(20)).astype(int)
        colors[nonnoise] = cm.tab20(hashed / 19.0)
    colors[~nonnoise] = (0.2, 0.2, 0.2, 0.18)
    colors[:, 3] = np.where(nonnoise, alpha, 0.18)
    return colors


def energy_colors(energy: np.ndarray, *, alpha: float) -> np.ndarray:
    if len(energy) == 0:
        return np.empty((0, 4), dtype=np.float32)
    norm = Normalize(
        vmin=float(np.quantile(energy, 0.05)),
        vmax=float(np.quantile(energy, 0.95)),
        clip=True,
    )
    colors = cm.plasma(norm(energy))
    colors[:, 3] = alpha
    return colors


def dense_voxel_payload(
    coords: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int, int]]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    shape = tuple((maxs - mins + 1).astype(int))
    filled = np.zeros(shape, dtype=bool)
    facecolors = np.zeros(shape + (4,), dtype=np.float32)
    rel = (coords - mins).astype(int)
    filled[rel[:, 0], rel[:, 1], rel[:, 2]] = True
    facecolors[rel[:, 0], rel[:, 1], rel[:, 2]] = colors
    x_edges = np.arange(int(mins[0]), int(maxs[0]) + 2)
    y_edges = np.arange(int(mins[1]), int(maxs[1]) + 2)
    z_edges = np.arange(0, int(maxs[2] - mins[2]) + 2)
    x_grid, y_grid, z_grid = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    return filled, facecolors, x_grid, y_grid, z_grid, (int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), int(maxs[2] - mins[2] + 1))


def set_equal_cube_axes(
    ax: plt.Axes,
    bounds: tuple[int, int, int, int, int],
    *,
    fixed_detector_xy: bool = False,
) -> None:
    xmin, xmax, ymin, ymax, z_span = bounds
    if fixed_detector_xy:
        xmin, xmax, ymin, ymax = 0, 255, 0, 255
    x_span = max(1, xmax - xmin + 1)
    y_span = max(1, ymax - ymin + 1)
    z_span = max(1, z_span)
    ax.set_box_aspect((x_span, y_span, z_span))
    ax.set_xlim(xmin, xmax + 1)
    ax.set_ylim(ymin, ymax + 1)
    ax.set_zlim(0, z_span)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel("fine time bin")
    ax.view_init(elev=23, azim=-56)


def draw_sparse_voxels(
    ax: plt.Axes,
    coords: np.ndarray,
    colors: np.ndarray,
    *,
    fixed_detector_xy: bool,
    z_axis_span: int | None = None,
) -> None:
    """Draw only occupied unit cubes, while preserving equal data scaling."""
    faces: list[list[tuple[float, float, float]]] = []
    facecolors: list[tuple[float, float, float, float]] = []
    for (x, y, z), color in zip(coords, colors, strict=False):
        x0 = float(x)
        x1 = x0 + 1.0
        y0 = float(y)
        y1 = y0 + 1.0
        z0 = float(z)
        z1 = z0 + 1.0
        cube_faces = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
        faces.extend(cube_faces)
        rgba = tuple(float(v) for v in color)
        facecolors.extend([rgba] * 6)
    collection = Poly3DCollection(
        faces,
        facecolors=facecolors,
        edgecolors=(0.0, 0.0, 0.0, 0.12),
        linewidths=0.12,
    )
    ax.add_collection3d(collection)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    z_span = int(z_axis_span) if z_axis_span is not None else int(maxs[2] - mins[2] + 1)
    bounds = (int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1]), z_span)
    set_equal_cube_axes(ax, bounds, fixed_detector_xy=fixed_detector_xy)


def render_window(
    vox: dict[str, np.ndarray],
    out_path: Path,
    *,
    z_start: int,
    z_end: int,
    source: str,
    rank: int,
    alpha: float,
    fixed_detector_xy: bool,
    plot_time_bin_factor: int,
    fixed_window_z: bool,
) -> dict[str, int | float | str]:
    coords = vox["coords"]
    labels = vox["labels"]
    raw_labels = vox["raw_labels"]
    if len(coords) == 0:
        raise ValueError("Cannot render an empty time window.")
    label_colors = stable_label_colors(labels, alpha=alpha)
    e_colors = energy_colors(vox["energy_mean"], alpha=alpha)

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    for ax, colors, title in [
        (axes[0], label_colors, "DBSCAN particle IDs"),
        (axes[1], e_colors, "energy"),
    ]:
        z_axis_span = None
        if fixed_window_z:
            z_axis_span = int(np.ceil((z_end - z_start) / plot_time_bin_factor))
        draw_sparse_voxels(ax, coords, colors, fixed_detector_xy=fixed_detector_xy, z_axis_span=z_axis_span)
        ax.set_title(title, pad=8)

    start_s = z_start * FINE_TIME_BIN_NS * 1e-9
    span_ns = (z_end - z_start) * FINE_TIME_BIN_NS
    plotted_bin_ns = plot_time_bin_factor * FINE_TIME_BIN_NS
    particle_labels = raw_labels[raw_labels >= 0]
    n_particles = len(np.unique(particle_labels))
    noise_fraction = float(np.mean(raw_labels < 0)) if len(raw_labels) else 0.0
    fig.suptitle(
        f"window {rank} | {Path(source).name} | t={start_s:.6f}s + {span_ns:.1f}ns | "
        f"z cube={plotted_bin_ns:.2f}ns | "
        f"{int(vox['raw_hit_count'][0])} hits | {n_particles} particles | noise {noise_fraction:.2%}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return {
        "rank": rank,
        "z_start": int(z_start),
        "z_end": int(z_end),
        "start_s": float(start_s),
        "span_ns": float(span_ns),
        "plot_time_bin_factor": int(plot_time_bin_factor),
        "plotted_bin_ns": float(plotted_bin_ns),
        "hits": int(vox["raw_hit_count"][0]),
        "occupied_voxels": int(len(coords)),
        "particles": int(n_particles),
        "noise_fraction": noise_fraction,
        "image": out_path.as_posix(),
    }


def plot_context(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    selected: list[tuple[int, int]],
    *,
    assets: Path,
    source: str,
) -> tuple[Path, Path, Path]:
    seconds = z.astype(np.float64) * FINE_TIME_BIN_NS * 1e-9
    selected_starts = [start for start, _ in selected]
    selected_width = selected[0][0] * 0 + (selected[1][0] - selected[0][0] if len(selected) > 1 else 0)
    _ = selected_width

    time_density = assets / "whole_file_time_density.png"
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.hist(seconds, bins=900, color="#334155", log=True)
    for start in selected_starts:
        ax.axvline(start * FINE_TIME_BIN_NS * 1e-9, color="#dc2626", alpha=0.7, linewidth=0.9)
    ax.set_title(f"Whole-file hit density over time: {Path(source).name}")
    ax.set_xlabel("relative acquisition time (s)")
    ax.set_ylabel("hits / bin (log)")
    fig.savefig(time_density, dpi=170)
    plt.close(fig)

    xy_density = assets / "whole_file_xy_density.png"
    hist, _, _ = np.histogram2d(x, y, bins=[np.arange(257), np.arange(257)])
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(hist.T + 1, origin="lower", cmap="magma", norm=LogNorm())
    ax.set_title("Whole-file XY occupancy, log scale")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.colorbar(im, ax=ax, label="hits + 1")
    fig.savefig(xy_density, dpi=170)
    plt.close(fig)

    particle_time = assets / "whole_file_particle_time_size.png"
    valid = labels >= 0
    particle_ids, starts, counts = np.unique(labels[valid], return_index=True, return_counts=True)
    z_valid = z[valid]
    start_s = z_valid[starts].astype(np.float64) * FINE_TIME_BIN_NS * 1e-9
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    sample = np.arange(len(particle_ids))
    if len(sample) > 80_000:
        rng = np.random.default_rng(20260505)
        sample = rng.choice(sample, size=80_000, replace=False)
    ax.scatter(start_s[sample], counts[sample], s=2, alpha=0.28, color="#2563eb", linewidths=0)
    ax.set_yscale("log")
    ax.set_title("DBSCAN particle size over file time")
    ax.set_xlabel("particle first-hit time (s)")
    ax.set_ylabel("hits per particle (log)")
    fig.savefig(particle_time, dpi=170)
    plt.close(fig)
    return time_density, xy_density, particle_time


def write_report(
    report: Path,
    *,
    title: str,
    npz_path: Path,
    source: str,
    total_hits: int,
    total_particles: int,
    noise_fraction: float,
    z_span_bins: int,
    context_paths: tuple[Path, Path, Path],
    rows: list[dict[str, int | float | str]],
    fixed_detector_xy: bool,
    select_by: str,
    plot_time_bin_factor: int,
    fixed_window_z: bool,
) -> None:
    rel_assets = context_paths[0].parent.relative_to(report.parent)
    lines = [
        f"# {title}",
        "",
        f"Source shard: `{npz_path}`",
        "",
        f"Raw source: `{source}`",
        "",
        "This is a whole-file inspection view. A literal one-image, equal-axis rendering of every cube is not useful "
        f"because this file spans `{z_span_bins:,}` fine-time bins. The report therefore uses whole-file summaries plus "
        "an equal-cube atlas of the densest short time windows.",
        "",
        f"- hits: `{total_hits:,}`",
        f"- particles: `{total_particles:,}`",
        f"- noise fraction: `{noise_fraction:.3f}`",
        f"- fine-time span: `{z_span_bins:,}` bins = `{z_span_bins * FINE_TIME_BIN_NS * 1e-9:.3f}` s",
        f"- plotted z cube: `{plot_time_bin_factor}` fine bins = `{plot_time_bin_factor * FINE_TIME_BIN_NS:.4f}` ns",
        "",
        "## Whole-File Context",
        "",
        f"![time density]({rel_assets / context_paths[0].name})",
        "",
        f"![xy density]({rel_assets / context_paths[1].name})",
        "",
        f"![particle size over time]({rel_assets / context_paths[2].name})",
        "",
        "## Densest Time Windows As Equal Cubes",
        "",
        "Each window uses real cube scaling: one x pixel, one y pixel, and one fine time bin have the same drawn size. "
        "The z axis is local to the selected time window, but the table gives absolute relative acquisition time.",
        "",
        (
            "The x/y axes are fixed to the full detector plane `0..256`."
            if fixed_detector_xy
            else "The x/y axes are zoomed to each window's occupied bounding box."
        ),
        (
            "The z axis is fixed to the full selected time slab."
            if fixed_window_z
            else "The z axis is zoomed to each window's occupied time span."
        ),
        "",
        f"Windows were selected by `{select_by}`.",
        "",
        "| rank | start s | span ns | z cube ns | hits | voxels | particles | noise frac | image |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        img = Path(str(row["image"]))
        lines.append(
            f"| {int(row['rank'])} | {float(row['start_s']):.6f} | {float(row['span_ns']):.1f} | "
            f"{float(row['plotted_bin_ns']):.4f} | "
            f"{int(row['hits']):,} | {int(row['occupied_voxels']):,} | {int(row['particles']):,} | "
            f"{float(row['noise_fraction']):.3f} | [png]({img.relative_to(report.parent)}) |"
        )
    for row in rows:
        img = Path(str(row["image"]))
        lines += [
            "",
            f"### Window {int(row['rank'])}",
            "",
            f"![window {int(row['rank'])}]({img.relative_to(report.parent)})",
        ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    data = load_hits(args.npz)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    energy = data["energy"]
    labels = data["labels"]
    source = str(data["source_path"])
    if args.window_start_fine_bin:
        selected = [(int(start), 0) for start in args.window_start_fine_bin]
    else:
        selected = select_dense_windows(
            z,
            labels,
            window_bins=args.window_fine_bins,
            count=args.windows,
            min_gap=args.min_window_gap,
            select_by=args.select_by,
        )
    context = plot_context(x, y, z, labels, selected, assets=args.assets, source=source)
    rows: list[dict[str, int | float | str]] = []
    for rank, (start, count) in enumerate(selected, start=1):
        end = start + args.window_fine_bins
        vox = voxelize_window(
            x,
            y,
            z,
            energy,
            labels,
            z_start=start,
            z_end=end,
            plot_time_bin_factor=args.plot_time_bin_factor,
        )
        image = args.assets / f"whole_file_window_{rank:02d}_z{start}_{end}_cubes.png"
        row = render_window(
            vox,
            image,
            z_start=start,
            z_end=end,
            source=source,
            rank=rank,
            alpha=args.alpha,
            fixed_detector_xy=args.fixed_detector_xy,
            plot_time_bin_factor=args.plot_time_bin_factor,
            fixed_window_z=args.fixed_window_z,
        )
        rows.append(row)
        print(f"rendered {image} ({count} hits in bucket)")
    import pandas as pd

    pd.DataFrame(rows).to_csv(args.assets / "whole_file_dense_window_summary.csv", index=False)
    total_particles = len(np.unique(labels[labels >= 0]))
    write_report(
        args.report,
        title=args.title,
        npz_path=args.npz,
        source=source,
        total_hits=len(labels),
        total_particles=total_particles,
        noise_fraction=float(np.mean(labels < 0)),
        z_span_bins=int(z.max() - z.min() + 1),
        context_paths=context,
        rows=rows,
        fixed_detector_xy=args.fixed_detector_xy,
        select_by=args.select_by,
        plot_time_bin_factor=args.plot_time_bin_factor,
        fixed_window_z=args.fixed_window_z,
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

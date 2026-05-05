#!/usr/bin/env python3
"""Render time windows whose particle count is close to a target value."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINE_TIME_BIN_NS = 25.0 / 16.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-particles", type=int, default=100)
    parser.add_argument("--tolerance", type=int, default=30)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--max-sources", type=int, default=24)
    parser.add_argument("--max-per-source", type=int, default=2)
    parser.add_argument("--time-scale", type=float, default=0.625)
    parser.add_argument(
        "--window-fine-bins",
        default="64000,128000,256000,512000,1024000,2048000",
        help="Comma-separated fine-time window sizes to try.",
    )
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--title", default="Aligned-Time eps5 Approx. 100-Particle Windows")
    return parser.parse_args()


def stable_label_values(labels: np.ndarray) -> np.ndarray:
    values = np.zeros(len(labels), dtype=np.float64)
    valid = labels >= 0
    if np.any(valid):
        values[valid] = ((labels[valid].astype(np.uint64) * np.uint64(2654435761)) % np.uint64(4096)).astype(float)
    values[~valid] = np.nan
    return values


def load_npz(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as data:
        time = data["hit_time"].astype(np.float64)
        return {
            "x": data["hit_x"].astype(np.float64),
            "y": data["hit_y"].astype(np.float64),
            "time": time,
            "z_fine": np.rint(time * 16.0).astype(np.int64),
            "energy": data["hit_energy"].astype(np.float64),
            "labels": data["hit_particle_id"].astype(np.int32),
            "source": str(data["source_path"]),
        }


def scan_source(
    data: dict[str, np.ndarray | str],
    *,
    source_path: str,
    npz_path: Path,
    window_fine_bins: list[int],
    target: int,
    tolerance: int,
) -> list[dict[str, int | float | str]]:
    z_fine = np.asarray(data["z_fine"])
    labels = np.asarray(data["labels"])
    rows: list[dict[str, int | float | str]] = []
    if len(z_fine) == 0:
        return rows
    for width in window_fine_bins:
        bucket = z_fine // width
        order = np.argsort(bucket, kind="stable")
        sorted_bucket = bucket[order]
        sorted_labels = labels[order]
        unique, starts, counts = np.unique(sorted_bucket, return_index=True, return_counts=True)
        for bucket_id, start, count in zip(unique, starts, counts, strict=False):
            chunk = sorted_labels[start : start + count]
            particles = int(len(np.unique(chunk[chunk >= 0])))
            if abs(particles - target) > tolerance:
                continue
            noise_fraction = float(np.mean(chunk < 0)) if count else 0.0
            # Prefer target closeness, then readable hit count, then low noise.
            score = abs(particles - target) + 0.00002 * count + 4.0 * noise_fraction
            rows.append(
                {
                    "score": float(score),
                    "source_path": source_path,
                    "npz_path": npz_path.as_posix(),
                    "window_fine_bins": int(width),
                    "z_start": int(bucket_id * width),
                    "z_end": int((bucket_id + 1) * width),
                    "hits": int(count),
                    "particles": particles,
                    "noise_fraction": noise_fraction,
                }
            )
    return rows


def select_windows(args: argparse.Namespace) -> list[dict[str, int | float | str]]:
    manifest = pd.read_csv(args.manifest)
    sources = (
        manifest[manifest["status"].eq("ok")]
        .sort_values(["particle_count", "row_count"], ascending=False)
        .head(args.max_sources)
    )
    widths = [int(item.strip()) for item in args.window_fine_bins.split(",") if item.strip()]
    candidates: list[dict[str, int | float | str]] = []
    for row in sources.itertuples(index=False):
        data = load_npz(Path(str(row.output_path)))
        candidates.extend(
            scan_source(
                data,
                source_path=str(row.source_path),
                npz_path=Path(str(row.output_path)),
                window_fine_bins=widths,
                target=args.target_particles,
                tolerance=args.tolerance,
            )
        )
    candidates.sort(key=lambda row: (float(row["score"]), -int(row["hits"])))

    selected: list[dict[str, int | float | str]] = []
    per_source: dict[str, int] = {}
    occupied: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for row in candidates:
        source = str(row["source_path"])
        if per_source.get(source, 0) >= args.max_per_source:
            continue
        key = (source, int(row["window_fine_bins"]))
        z0 = int(row["z_start"])
        z1 = int(row["z_end"])
        overlaps = any(max(z0, lo) < min(z1, hi) for lo, hi in occupied.get(key, []))
        if overlaps:
            continue
        selected.append(row)
        per_source[source] = per_source.get(source, 0) + 1
        occupied.setdefault(key, []).append((z0, z1))
        if len(selected) >= args.windows:
            break
    return selected


def configure_axes(ax, z_span: float) -> None:
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_zlim(0, z_span)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel("DBSCAN-scaled time")
    ax.set_box_aspect((1.0, 1.0, 0.65))
    ax.view_init(elev=23, azim=-58)


def render_window(row: dict[str, int | float | str], *, rank: int, args: argparse.Namespace) -> dict[str, int | float | str]:
    data = load_npz(Path(str(row["npz_path"])))
    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    time = np.asarray(data["time"])
    z_fine = np.asarray(data["z_fine"])
    energy = np.asarray(data["energy"])
    labels = np.asarray(data["labels"])
    z0 = int(row["z_start"])
    z1 = int(row["z_end"])
    mask = (z_fine >= z0) & (z_fine < z1)
    wx = x[mask]
    wy = y[mask]
    wz = (time[mask] - (z0 / 16.0)) / args.time_scale
    we = energy[mask]
    wl = labels[mask]
    valid = wl >= 0
    z_span = (z1 - z0) / (16.0 * args.time_scale)

    image = args.assets / f"approx100_window_{rank:02d}_{Path(str(row['source_path'])).stem}_z{z0}_{z1}.png"
    image.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 6.8), constrained_layout=True)
    ax_labels = fig.add_subplot(1, 2, 1, projection="3d")
    ax_energy = fig.add_subplot(1, 2, 2, projection="3d")

    ax_labels.scatter(wx[~valid], wy[~valid], wz[~valid], c="lightgray", s=args.point_size, alpha=0.22, linewidths=0)
    ax_labels.scatter(
        wx[valid],
        wy[valid],
        wz[valid],
        c=stable_label_values(wl)[valid],
        cmap="tab20",
        s=args.point_size,
        alpha=0.86,
        linewidths=0,
    )
    ax_labels.set_title("DBSCAN particle IDs")

    sc = ax_energy.scatter(wx, wy, wz, c=we, cmap="plasma", s=args.point_size, alpha=0.86, linewidths=0)
    ax_energy.set_title("energy")
    fig.colorbar(sc, ax=ax_energy, shrink=0.70, label="log1p(ToT)")
    configure_axes(ax_labels, z_span)
    configure_axes(ax_energy, z_span)

    span_us = (z1 - z0) * FINE_TIME_BIN_NS / 1000.0
    fig.suptitle(
        f"rank {rank} | {row['source_path']} | {int(row['hits']):,} hits | "
        f"{int(row['particles'])} particles | {span_us:.1f} us | window {int(row['window_fine_bins']):,} fine bins",
        fontsize=10.5,
    )
    fig.savefig(image, dpi=170)
    plt.close(fig)
    out = dict(row)
    out.update(
        {
            "rank": rank,
            "span_us": float(span_us),
            "scaled_time_span": float(z_span),
            "image": image.as_posix(),
        }
    )
    return out


def write_report(args: argparse.Namespace, rows: list[dict[str, int | float | str]]) -> None:
    lines = [
        f"# {args.title}",
        "",
        f"Source manifest: `{args.manifest}`",
        "",
        f"Target particle count: `{args.target_particles}` with tolerance `{args.tolerance}`.",
        "",
        f"These are fixed time buckets selected from the highest-particle-count shards. "
        f"The z coordinate is `z = (ToA - FToA / 16 - window_start) / {args.time_scale:g}`, "
        "the same scaled time used by the current aligned-time DBSCAN setting.",
        "",
        "| rank | source | start s | span us | fine bins | scaled span | hits | particles | noise frac | image |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        image = Path(str(row["image"]))
        start_s = int(row["z_start"]) * FINE_TIME_BIN_NS * 1e-9
        lines.append(
            f"| {int(row['rank'])} | `{row['source_path']}` | {start_s:.6f} | "
            f"{float(row['span_us']):.1f} | {int(row['window_fine_bins']):,} | "
            f"{float(row['scaled_time_span']):.1f} | {int(row['hits']):,} | "
            f"{int(row['particles'])} | {float(row['noise_fraction']):.3f} | "
            f"[png]({image.relative_to(args.report.parent)}) |"
        )
    for row in rows:
        image = Path(str(row["image"]))
        lines += [
            "",
            f"### Window {int(row['rank'])}",
            "",
            f"![window {int(row['rank'])}]({image.relative_to(args.report.parent)})",
        ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")


def main() -> None:
    args = parse_args()
    selected = select_windows(args)
    if not selected:
        raise SystemExit("No matching windows found. Try increasing --tolerance or --max-sources.")
    args.assets.mkdir(parents=True, exist_ok=True)
    rendered = [render_window(row, rank=rank, args=args) for rank, row in enumerate(selected, start=1)]
    pd.DataFrame(rendered).to_csv(args.assets / "approx_100_particle_windows.csv", index=False)
    write_report(args, rendered)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render DBSCAN windows using the same scaled-time coordinate as clustering."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--time-scale", type=float, default=0.625)
    parser.add_argument("--window-fine-bins", type=int, default=512_000)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--min-particles", type=int, default=11)
    parser.add_argument("--fixed-detector-xy", action="store_true")
    parser.add_argument("--title", default="Scaled-Time DBSCAN Many-Particle Gallery")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "x": data["hit_x"].astype(np.float64),
            "y": data["hit_y"].astype(np.float64),
            "time": data["hit_time"].astype(np.float64),
            "z_fine": np.rint(data["hit_time"].astype(np.float64) * 16.0).astype(np.int64),
            "energy": data["hit_energy"].astype(np.float64),
            "labels": data["hit_particle_id"].astype(np.int32),
            "source": str(data["source_path"]),
        }


def select_windows(
    z_fine: np.ndarray,
    labels: np.ndarray,
    *,
    window_fine_bins: int,
    windows: int,
    min_particles: int,
) -> list[dict[str, int | float]]:
    bucket = z_fine // window_fine_bins
    order = np.argsort(bucket, kind="stable")
    sorted_bucket = bucket[order]
    sorted_labels = labels[order]
    unique, starts, counts = np.unique(sorted_bucket, return_index=True, return_counts=True)
    rows = []
    for bucket_id, start, count in zip(unique, starts, counts, strict=False):
        chunk = sorted_labels[start : start + count]
        particles = len(np.unique(chunk[chunk >= 0]))
        if particles < min_particles:
            continue
        rows.append(
            {
                "z_start": int(bucket_id * window_fine_bins),
                "z_end": int((bucket_id + 1) * window_fine_bins),
                "hits": int(count),
                "particles": int(particles),
                "noise_fraction": float(np.mean(chunk < 0)),
            }
        )
    rows.sort(key=lambda row: (int(row["particles"]), int(row["hits"])), reverse=True)
    return rows[:windows]


def stable_label_values(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.float64)
    valid = labels >= 0
    if np.any(valid):
        hashed = ((labels[valid].astype(np.uint64) * np.uint64(2654435761)) % np.uint64(4096)).astype(float)
        out[valid] = hashed
    out[~valid] = np.nan
    return out


def render_window(
    data: dict[str, np.ndarray | str],
    row: dict[str, int | float],
    *,
    time_scale: float,
    fixed_detector_xy: bool,
    out_path: Path,
    rank: int,
) -> dict[str, int | float | str]:
    x = data["x"]
    y = data["y"]
    time = data["time"]
    z_fine = data["z_fine"]
    labels = data["labels"]
    energy = data["energy"]
    source = str(data["source"])
    z_start = int(row["z_start"])
    z_end = int(row["z_end"])
    mask = (z_fine >= z_start) & (z_fine < z_end)
    if not np.any(mask):
        raise ValueError(f"Selected empty window {z_start}..{z_end}")

    wx = x[mask]
    wy = y[mask]
    wt_scaled = (time[mask] - (z_start / 16.0)) / time_scale
    wlabels = labels[mask]
    wenergy = energy[mask]
    valid = wlabels >= 0

    fig = plt.figure(figsize=(15, 6.8), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]

    label_values = stable_label_values(wlabels)
    axes[0].scatter(wx[~valid], wy[~valid], wt_scaled[~valid], c="lightgray", s=9, alpha=0.24, linewidths=0)
    axes[0].scatter(
        wx[valid],
        wy[valid],
        wt_scaled[valid],
        c=label_values[valid],
        cmap="tab20",
        s=12,
        alpha=0.86,
        linewidths=0,
    )
    axes[0].set_title("DBSCAN particle IDs")

    norm = Normalize(
        vmin=float(np.quantile(wenergy, 0.05)),
        vmax=float(np.quantile(wenergy, 0.95)),
        clip=True,
    )
    energy_colors = cm.plasma(norm(wenergy))
    axes[1].scatter(wx, wy, wt_scaled, c=energy_colors, s=12, alpha=0.86, linewidths=0)
    axes[1].set_title("energy")

    z_span_scaled = float((z_end - z_start) / (16.0 * time_scale))
    for ax in axes:
        if fixed_detector_xy:
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)
        else:
            ax.set_xlim(float(wx.min()) - 2, float(wx.max()) + 2)
            ax.set_ylim(float(wy.min()) - 2, float(wy.max()) + 2)
        ax.set_zlim(0, z_span_scaled)
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
        ax.set_zlabel(f"scaled time / {time_scale:g}")
        # Use a readable display aspect while the z tick values remain DBSCAN-scaled units.
        ax.set_box_aspect((1.0, 1.0, 0.72))
        ax.view_init(elev=23, azim=-58)

    fig.suptitle(
        f"rank {rank} | {Path(source).name} | "
        f"{int(row['hits'])} hits | {int(row['particles'])} particles | "
        f"{(z_end - z_start) * FINE_TIME_BIN_NS / 1000.0:.1f} us window | "
        f"z is DBSCAN-scaled time",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return {
        **row,
        "rank": rank,
        "span_us": float((z_end - z_start) * FINE_TIME_BIN_NS / 1000.0),
        "scaled_time_span": z_span_scaled,
        "image": out_path.as_posix(),
    }


def write_report(
    *,
    report: Path,
    title: str,
    npz_path: Path,
    source: str,
    rows: list[dict[str, int | float | str]],
    time_scale: float,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"Source shard: `{npz_path}`",
        "",
        f"Raw source: `{source}`",
        "",
        f"These plots use DBSCAN-scaled time on z: `z = (ToA - FToA / 16 - window_start) / {time_scale:g}`.",
        "The visual z aspect is compressed for readability, but the z-axis tick values are the scaled-time values used by DBSCAN.",
        "",
        "| rank | start s | span us | scaled time span | hits | particles | noise frac | image |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        img = Path(str(row["image"]))
        start_s = float(int(row["z_start"]) * FINE_TIME_BIN_NS * 1e-9)
        lines.append(
            f"| {int(row['rank'])} | {start_s:.6f} | {float(row['span_us']):.1f} | "
            f"{float(row['scaled_time_span']):.1f} | {int(row['hits']):,} | "
            f"{int(row['particles'])} | {float(row['noise_fraction']):.3f} | "
            f"[png]({img.relative_to(report.parent)}) |"
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
    data = load_npz(args.npz)
    selected = select_windows(
        data["z_fine"],
        data["labels"],
        window_fine_bins=args.window_fine_bins,
        windows=args.windows,
        min_particles=args.min_particles,
    )
    if not selected:
        raise SystemExit("No windows matched the requested particle count.")
    rendered = []
    for rank, row in enumerate(selected, start=1):
        image = args.assets / f"scaled_time_window_{rank:02d}_z{int(row['z_start'])}_{int(row['z_end'])}.png"
        rendered.append(
            render_window(
                data,
                row,
                time_scale=args.time_scale,
                fixed_detector_xy=args.fixed_detector_xy,
                out_path=image,
                rank=rank,
            )
        )
        print(f"rendered {image}")
    args.assets.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rendered).to_csv(args.assets / "scaled_time_windows.csv", index=False)
    write_report(
        report=args.report,
        title=args.title,
        npz_path=args.npz,
        source=str(data["source"]),
        rows=rendered,
        time_scale=args.time_scale,
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

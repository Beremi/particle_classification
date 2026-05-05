#!/usr/bin/env python3
"""Render the longest-duration particles from NPZ particle shards."""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIME_TICK_NS = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--candidates-per-file", type=int, default=12)
    parser.add_argument("--time-scale", type=float, default=0.625)
    parser.add_argument("--title", default="Aligned-Time eps5 Long-Timespan Particles")
    return parser.parse_args()


def scan_longest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest)
    heap: list[tuple[float, str, str, int, int, float, float, float, float, float, float, float, int, int]] = []
    for row in manifest.itertuples(index=False):
        if getattr(row, "status", "") != "ok" or int(row.particle_count) <= 0:
            continue
        shard = Path(str(row.output_path))
        with np.load(shard, allow_pickle=False) as data:
            t_min = data["particle_time_min"].astype(np.float64)
            t_max = data["particle_time_max"].astype(np.float64)
            n_hits = data["particle_n_hits"].astype(np.int64)
            particle_id = data["particle_id"].astype(np.int64)
            energy_sum = data["particle_energy_sum"].astype(np.float64)
            x_min = data["particle_x_min"].astype(np.float64)
            x_max = data["particle_x_max"].astype(np.float64)
            y_min = data["particle_y_min"].astype(np.float64)
            y_max = data["particle_y_max"].astype(np.float64)
            offsets = data["particle_offsets"].astype(np.int64)
            span = t_max - t_min
            keep = min(args.candidates_per_file, len(span))
            if keep <= 0:
                continue
            for i in np.argpartition(span, -keep)[-keep:]:
                record = (
                    float(span[i]),
                    str(row.source_path),
                    str(shard),
                    int(particle_id[i]),
                    int(n_hits[i]),
                    float(energy_sum[i]),
                    float(t_min[i]),
                    float(t_max[i]),
                    float(x_min[i]),
                    float(x_max[i]),
                    float(y_min[i]),
                    float(y_max[i]),
                    int(offsets[i]),
                    int(offsets[i + 1]),
                )
                if len(heap) < max(args.limit * 4, args.limit):
                    heapq.heappush(heap, record)
                elif record[0] > heap[0][0]:
                    heapq.heapreplace(heap, record)
    columns = [
        "time_span_ticks",
        "source_path",
        "shard",
        "particle_id",
        "n_hits",
        "energy_sum",
        "time_min",
        "time_max",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "start",
        "end",
    ]
    df = pd.DataFrame(sorted(heap, reverse=True), columns=columns).head(args.limit).copy()
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["time_span_ns"] = df["time_span_ticks"] * TIME_TICK_NS
    df["scaled_time_span"] = df["time_span_ticks"] / args.time_scale
    return df


def set_particle_axes(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    pad = 2.0
    xmin, xmax = float(np.min(x) - pad), float(np.max(x) + pad)
    ymin, ymax = float(np.min(y) - pad), float(np.max(y) + pad)
    zmin, zmax = float(np.min(z) - pad), float(np.max(z) + pad)
    xspan = max(1.0, xmax - xmin)
    yspan = max(1.0, ymax - ymin)
    zspan = max(1.0, zmax - zmin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((xspan, yspan, zspan))
    ax.view_init(elev=24, azim=-58)


def render_particle(row: pd.Series, args: argparse.Namespace) -> dict[str, object]:
    with np.load(Path(row["shard"]), allow_pickle=False) as data:
        start = int(row["start"])
        end = int(row["end"])
        x = data["hit_x"][start:end].astype(np.float64)
        y = data["hit_y"][start:end].astype(np.float64)
        time = data["hit_time"][start:end].astype(np.float64)
        energy = data["hit_energy"][start:end].astype(np.float64)

    order = np.argsort(time, kind="mergesort")
    x = x[order]
    y = y[order]
    time = time[order]
    energy = energy[order]
    z_scaled = (time - float(np.min(time))) / args.time_scale
    t_ns = (time - float(np.min(time))) * TIME_TICK_NS

    source_stem = Path(str(row["source_path"])).stem
    image = args.assets / f"long_timespan_{int(row['rank']):02d}_{source_stem}_p{int(row['particle_id'])}.png"
    image.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    scatter = ax3d.scatter(x, y, z_scaled, c=energy, cmap="plasma", s=14, alpha=0.92, linewidths=0)
    ax3d.plot(x, y, z_scaled, color="black", alpha=0.24, linewidth=0.8)
    set_particle_axes(ax3d, x, y, z_scaled)
    ax3d.set_xlabel("x pixel")
    ax3d.set_ylabel("y pixel")
    ax3d.set_zlabel(f"scaled time / {args.time_scale:g}")
    ax3d.set_title("3D path, energy-colored")
    fig.colorbar(scatter, ax=ax3d, shrink=0.70, label="log1p(ToT)")

    ax2d = fig.add_subplot(1, 2, 2)
    sc2 = ax2d.scatter(x, y, c=t_ns, cmap="viridis", s=18, alpha=0.9, linewidths=0)
    ax2d.plot(x, y, color="black", alpha=0.22, linewidth=0.8)
    ax2d.set_xlim(max(-2, float(np.min(x) - 3)), min(258, float(np.max(x) + 3)))
    ax2d.set_ylim(max(-2, float(np.min(y) - 3)), min(258, float(np.max(y) + 3)))
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("x pixel")
    ax2d.set_ylabel("y pixel")
    ax2d.set_title("XY projection, colored by elapsed time")
    fig.colorbar(sc2, ax=ax2d, shrink=0.82, label="elapsed ns")

    fig.suptitle(
        f"rank {int(row['rank'])} | {row['source_path']} | particle {int(row['particle_id'])} | "
        f"{int(row['n_hits'])} hits | span {float(row['time_span_ns']):.1f} ns",
        fontsize=11,
    )
    fig.savefig(image, dpi=180)
    plt.close(fig)

    record = row.to_dict()
    record["image"] = image.as_posix()
    return record


def write_report(args: argparse.Namespace, rows: list[dict[str, object]], manifest_summary: dict[str, object]) -> None:
    rel_assets = args.assets.relative_to(args.report.parent)
    lines = [
        f"# {args.title}",
        "",
        "This report uses the full all-file reestimate from `local_data/processed/particles_aligned_time_eps5_v001/`.",
        "Ranking is by per-particle `particle_time_max - particle_time_min` using corrected `ToA - FToA / 16` time.",
        "",
        "The 3D panels use DBSCAN-scaled time on z: `(hit_time - particle_time_min) / 0.625`. "
        "The XY panels are colored by elapsed nanoseconds.",
        "",
        "## Reestimate Summary",
        "",
        f"- Files: `{manifest_summary['files']}`",
        f"- Raw hits: `{int(manifest_summary['hits']):,}`",
        f"- Particles: `{int(manifest_summary['particles']):,}`",
        f"- Weighted noise fraction: `{float(manifest_summary['noise_fraction']):.3f}`",
        f"- Slowest file runtime: `{float(manifest_summary['slowest_runtime_s']):.3f} s`",
        "",
        "## Longest Timespan Particles",
        "",
        "| rank | source | particle | hits | span ns | span ticks | scaled span | energy sum | image |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        image = Path(str(row["image"]))
        lines.append(
            f"| {int(row['rank'])} | `{row['source_path']}` | {int(row['particle_id'])} | "
            f"{int(row['n_hits']):,} | {float(row['time_span_ns']):.1f} | "
            f"{float(row['time_span_ticks']):.4f} | {float(row['scaled_time_span']):.2f} | "
            f"{float(row['energy_sum']):.1f} | [png]({image.relative_to(args.report.parent)}) |"
        )
    for row in rows:
        image = Path(str(row["image"]))
        lines += [
            "",
            f"### Rank {int(row['rank'])}",
            "",
            f"![rank {int(row['rank'])}]({image.relative_to(args.report.parent)})",
        ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    print(f"assets in {rel_assets}")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    df = scan_longest(args)
    args.assets.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.assets / "longest_timespan_particles.csv", index=False)
    rows = [render_particle(row, args) for _, row in df.iterrows()]
    pd.DataFrame(rows).to_csv(args.assets / "longest_timespan_particles_rendered.csv", index=False)
    summary = {
        "files": len(manifest),
        "hits": int(manifest["row_count"].sum()),
        "particles": int(manifest["particle_count"].sum()),
        "noise_fraction": float(manifest["noise_count"].sum() / max(1, manifest["row_count"].sum())),
        "slowest_runtime_s": float(manifest["runtime_s"].max()),
    }
    write_report(args, rows, summary)


if __name__ == "__main__":
    main()

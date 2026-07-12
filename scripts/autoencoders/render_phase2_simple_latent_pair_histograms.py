#!/usr/bin/env python3
"""Render all pairwise 2D histograms for the simple z8 voxel-AE latent space."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_percentile_bounds(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.nanmean(finite)) if finite.size else 0.0
        return center - 1.0, center + 1.0
    margin = 0.03 * float(hi - lo)
    return float(lo - margin), float(hi + margin)


def plot_hist2d(
    z: np.ndarray,
    i: int,
    j: int,
    bounds: list[tuple[float, float]],
    bins: int,
    path: Path,
    title_prefix: str,
) -> dict[str, object]:
    xi = z[:, i]
    yj = z[:, j]
    x_range = bounds[i]
    y_range = bounds[j]
    hist, xedges, yedges = np.histogram2d(xi, yj, bins=bins, range=[x_range, y_range])
    corr = float(np.corrcoef(xi, yj)[0, 1])

    fig, ax = plt.subplots(figsize=(6.5, 5.6), dpi=160)
    mesh = ax.imshow(
        np.log1p(hist.T),
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="magma",
    )
    ax.set_xlabel(f"z{i}")
    ax.set_ylabel(f"z{j}")
    ax.set_title(f"{title_prefix}: z{i} vs z{j}  (corr {corr:+.3f})")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log(1 + particle count)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return {
        "dim_i": i,
        "dim_j": j,
        "corr": corr,
        "x_min": x_range[0],
        "x_max": x_range[1],
        "y_min": y_range[0],
        "y_max": y_range[1],
        "nonempty_bins": int(np.count_nonzero(hist)),
        "max_bin_count": int(hist.max()) if hist.size else 0,
        "image": path.as_posix(),
    }


def plot_contact_sheet(image_paths: list[Path], labels: list[str], path: Path) -> None:
    n = len(image_paths)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.1, rows * 3.4), dpi=150)
    flat_axes = np.asarray(axes).ravel()
    for ax, img_path, label in zip(flat_axes, image_paths, labels, strict=False):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.set_axis_off()
    for ax in flat_axes[n:]:
        ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def plot_corr_matrix(corr: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=170)
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(corr.shape[0]), [f"z{i}" for i in range(corr.shape[0])])
    ax.set_yticks(range(corr.shape[0]), [f"z{i}" for i in range(corr.shape[0])])
    ax.set_title("Latent dimension correlation matrix")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-npz",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_latent_groups_v001/latents_and_groups_k32.npz"),
    )
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/latent_analysis/phase2-simple-z8-latent-pair-histograms.md"))
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("experimental_notes/assets/phase2_simple_z8_latent_pair_histograms_v001"),
    )
    parser.add_argument(
        "--local-out",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_latent_pair_histograms_v001"),
    )
    parser.add_argument("--space", choices=("z_shape", "z_norm"), default="z_shape")
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=99.8)
    args = parser.parse_args()

    ensure_dir(args.asset_dir)
    ensure_dir(args.local_out)
    ensure_dir(args.out.parent)

    data = np.load(args.latent_npz)
    z = np.asarray(data[args.space], dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 8:
        raise ValueError(f"expected an N x 8 latent array for {args.space}, got {z.shape}")

    bounds = [finite_percentile_bounds(z[:, i], args.clip_low, args.clip_high) for i in range(z.shape[1])]
    corr = np.corrcoef(z, rowvar=False)

    pair_rows: list[dict[str, object]] = []
    image_paths: list[Path] = []
    labels: list[str] = []
    pair_dir = args.asset_dir / "pairs"
    ensure_dir(pair_dir)
    for i, j in combinations(range(z.shape[1]), 2):
        img_path = pair_dir / f"z{i}_z{j}_hist2d.png"
        row = plot_hist2d(z, i, j, bounds, args.bins, img_path, args.space)
        pair_rows.append(row)
        image_paths.append(img_path)
        labels.append(f"z{i} vs z{j}")

    contact_sheet = args.asset_dir / "all_latent_pair_histograms.png"
    corr_path = args.asset_dir / "latent_correlation_matrix.png"
    plot_contact_sheet(image_paths, labels, contact_sheet)
    plot_corr_matrix(corr, corr_path)

    stats_path = args.local_out / f"{args.space}_pair_histogram_stats.csv"
    write_csv(
        stats_path,
        pair_rows,
        ["dim_i", "dim_j", "corr", "x_min", "x_max", "y_min", "y_max", "nonempty_bins", "max_bin_count", "image"],
    )

    rel_contact = contact_sheet.relative_to(args.out.parent).as_posix()
    rel_corr = corr_path.relative_to(args.out.parent).as_posix()
    lines = [
        "# Simple z8 Latent Pair Histograms",
        "",
        f"Latent source: `{args.latent_npz.as_posix()}`",
        f"Latent array: `{args.space}`",
        "",
        "This report shows every 2D pair projection of the 8D autoencoder latent space as a density histogram. "
        "The color scale is `log(1 + particle count)`, so both dense cores and tails remain visible.",
        "",
        "## Summary",
        "",
        "| item | value |",
        "|---|---:|",
        f"| particles | {z.shape[0]:,} |",
        f"| latent dimensions | {z.shape[1]} |",
        f"| pair histograms | {len(pair_rows)} |",
        f"| bins per axis | {args.bins} |",
        f"| plot bounds | p{args.clip_low:g} to p{args.clip_high:g} per dimension |",
        "",
        "## Overview",
        "",
        f"![all pair histograms]({rel_contact})",
        "",
        "## Correlations",
        "",
        f"![latent correlation matrix]({rel_corr})",
        "",
        "## Pair Gallery",
        "",
    ]
    for row in pair_rows:
        i = int(row["dim_i"])
        j = int(row["dim_j"])
        rel_img = Path(str(row["image"])).relative_to(args.out.parent).as_posix()
        lines += [
            f"### z{i} vs z{j}",
            "",
            f"- correlation: `{float(row['corr']):+.4f}`",
            f"- nonempty histogram bins: `{int(row['nonempty_bins']):,}`",
            f"- max bin count: `{int(row['max_bin_count']):,}`",
            "",
            f"![z{i} z{j} histogram]({rel_img})",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- These are model latent dimensions, not PCA axes.",
        "- Bounds are percentile-clipped for visualization only; the full arrays remain saved in the local NPZ artifact.",
        "- If you want clustering geometry, use `z_norm`; if you want the model's raw code values, use `z_shape`.",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()

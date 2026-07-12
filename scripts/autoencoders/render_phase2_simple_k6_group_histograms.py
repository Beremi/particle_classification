#!/usr/bin/env python3
"""Render K=6 latent groups and per-group pairwise histograms for the simple z8 AE."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile_bounds(z: np.ndarray, low: float, high: float) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for i in range(z.shape[1]):
        vals = z[:, i]
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [low, high])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            center = float(np.nanmean(vals)) if vals.size else 0.0
            lo, hi = center - 1.0, center + 1.0
        margin = 0.03 * float(hi - lo)
        bounds.append((float(lo - margin), float(hi + margin)))
    return bounds


def plot_group_sizes(labels: np.ndarray, out: Path) -> None:
    counts = np.bincount(labels)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=160)
    ax.bar(np.arange(len(counts)), counts, color="#4477aa")
    ax.set_xlabel("KMeans group")
    ax.set_ylabel("particles")
    ax.set_title("K=6 latent group sizes")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8, rotation=45)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_pca_groups(z_norm: np.ndarray, labels: np.ndarray, out: Path, sample: int, seed: int) -> np.ndarray:
    pca = PCA(n_components=2, random_state=seed)
    pc = pca.fit_transform(z_norm)
    rng = np.random.default_rng(seed)
    idx = np.arange(pc.shape[0])
    if idx.size > sample:
        idx = rng.choice(idx, size=sample, replace=False)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=170)
    sc = ax.scatter(pc[idx, 0], pc[idx, 1], c=labels[idx], s=2.0, cmap="tab10", alpha=0.65, linewidths=0)
    ax.set_xlabel("PC1 of z_norm")
    ax.set_ylabel("PC2 of z_norm")
    ax.set_title("K=6 groups in latent PCA plane")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("KMeans group")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return pc


def plot_pair_hist_on_axis(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bins: int,
    title: str,
) -> tuple[int, int]:
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])
    ax.imshow(
        np.log1p(hist.T),
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="magma",
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(title.split(" vs ")[0], fontsize=7)
    ax.set_ylabel(title.split(" vs ")[1], fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    return int(np.count_nonzero(hist)), int(hist.max()) if hist.size else 0


def plot_group_contact_sheet(
    z_shape: np.ndarray,
    mask: np.ndarray,
    bounds: list[tuple[float, float]],
    group: int,
    bins: int,
    out: Path,
) -> list[dict[str, object]]:
    pairs = list(combinations(range(z_shape.shape[1]), 2))
    cols = 4
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 2.75), dpi=150)
    flat_axes = np.asarray(axes).ravel()
    group_z = z_shape[mask]
    rows_out: list[dict[str, object]] = []
    for ax, (i, j) in zip(flat_axes, pairs, strict=False):
        nonempty, max_bin = plot_pair_hist_on_axis(
            ax,
            group_z[:, i],
            group_z[:, j],
            bounds[i],
            bounds[j],
            bins,
            f"z{i} vs z{j}",
        )
        corr = float(np.corrcoef(group_z[:, i], group_z[:, j])[0, 1]) if group_z.shape[0] > 2 else float("nan")
        rows_out.append(
            {
                "group": group,
                "dim_i": i,
                "dim_j": j,
                "count": int(group_z.shape[0]),
                "corr": corr,
                "nonempty_bins": nonempty,
                "max_bin_count": max_bin,
            }
        )
    for ax in flat_axes[len(pairs) :]:
        ax.set_axis_off()
    fig.suptitle(f"K=6 group {group}: all z_shape pair histograms, n={group_z.shape[0]:,}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out)
    plt.close(fig)
    return rows_out


def plot_group_1d_hists(z_shape: np.ndarray, labels: np.ndarray, out: Path, bins: int) -> None:
    groups = np.unique(labels)
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), dpi=160)
    flat = axes.ravel()
    for dim, ax in enumerate(flat):
        lo, hi = np.percentile(z_shape[:, dim], [0.2, 99.8])
        for group in groups:
            vals = z_shape[labels == group, dim]
            ax.hist(vals, bins=bins, range=(lo, hi), histtype="step", density=True, linewidth=1.1, label=f"g{group}")
        ax.set_title(f"z{dim}")
        ax.set_ylabel("density")
    flat[-1].legend(ncol=3, fontsize=8)
    fig.suptitle("Per-dimension z_shape histograms by K=6 group", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-npz",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_latent_groups_v001/latents_and_groups_k32.npz"),
    )
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/latent_analysis/phase2-simple-z8-k6-group-histograms.md"))
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("experimental_notes/assets/phase2_simple_z8_k6_group_histograms_v001"),
    )
    parser.add_argument(
        "--local-out",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_k6_group_histograms_v001"),
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--plot-sample", type=int, default=120000)
    parser.add_argument("--cluster-batch-size", type=int, default=20000)
    args = parser.parse_args()

    ensure_dir(args.out.parent)
    ensure_dir(args.asset_dir)
    ensure_dir(args.local_out)
    group_dir = args.asset_dir / "groups"
    ensure_dir(group_dir)

    data = np.load(args.latent_npz)
    z_shape = np.asarray(data["z_shape"], dtype=np.float32)
    z_norm = np.asarray(data["z_norm"], dtype=np.float32)
    if z_shape.shape != z_norm.shape or z_shape.ndim != 2:
        raise ValueError(f"bad latent shapes: z_shape={z_shape.shape}, z_norm={z_norm.shape}")

    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=args.cluster_batch_size,
        n_init=10,
        random_state=args.seed,
        reassignment_ratio=0.002,
    )
    labels = kmeans.fit_predict(z_norm).astype(np.int32)
    counts = np.bincount(labels, minlength=args.k)

    np.savez_compressed(
        args.local_out / f"kmeans_k{args.k}_labels.npz",
        labels=labels,
        centers=kmeans.cluster_centers_.astype(np.float32),
        inertia=np.asarray([kmeans.inertia_], dtype=np.float32),
    )

    bounds = percentile_bounds(z_shape, 0.2, 99.8)
    size_plot = args.asset_dir / "k6_group_sizes.png"
    pca_plot = args.asset_dir / "k6_pca_groups.png"
    one_d_plot = args.asset_dir / "k6_latent_1d_hists.png"
    plot_group_sizes(labels, size_plot)
    plot_pca_groups(z_norm, labels, pca_plot, args.plot_sample, args.seed)
    plot_group_1d_hists(z_shape, labels, one_d_plot, bins=90)

    pair_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for group in range(args.k):
        mask = labels == group
        img = group_dir / f"group_{group:02d}_all_pair_histograms.png"
        pair_rows.extend(plot_group_contact_sheet(z_shape, mask, bounds, group, args.bins, img))
        group_z = z_shape[mask]
        group_rows.append(
            {
                "group": group,
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "z0_mean": float(group_z[:, 0].mean()),
                "z1_mean": float(group_z[:, 1].mean()),
                "z2_mean": float(group_z[:, 2].mean()),
                "z3_mean": float(group_z[:, 3].mean()),
                "z4_mean": float(group_z[:, 4].mean()),
                "z5_mean": float(group_z[:, 5].mean()),
                "z6_mean": float(group_z[:, 6].mean()),
                "z7_mean": float(group_z[:, 7].mean()),
            }
        )

    write_csv(
        args.local_out / f"kmeans_k{args.k}_group_summary.csv",
        group_rows,
        ["group", "count", "fraction", "z0_mean", "z1_mean", "z2_mean", "z3_mean", "z4_mean", "z5_mean", "z6_mean", "z7_mean"],
    )
    write_csv(
        args.local_out / f"kmeans_k{args.k}_pair_histogram_stats.csv",
        pair_rows,
        ["group", "dim_i", "dim_j", "count", "corr", "nonempty_bins", "max_bin_count"],
    )

    rel_size = size_plot.relative_to(args.out.parent).as_posix()
    rel_pca = pca_plot.relative_to(args.out.parent).as_posix()
    rel_1d = one_d_plot.relative_to(args.out.parent).as_posix()
    lines = [
        "# Simple z8 K=6 Latent Group Histograms",
        "",
        f"Latent source: `{args.latent_npz.as_posix()}`",
        "",
        "KMeans was fitted on standardized `z_norm`, matching the previous latent grouping convention. "
        "The histograms below show raw model latent values `z_shape`, so you can inspect what each coarse group occupies in the actual encoded space.",
        "",
        "## Summary",
        "",
        "| item | value |",
        "|---|---:|",
        f"| particles | {z_shape.shape[0]:,} |",
        f"| latent dimensions | {z_shape.shape[1]} |",
        f"| KMeans groups | {args.k} |",
        f"| pair histograms per group | 28 |",
        f"| histogram bins per axis | {args.bins} |",
        f"| KMeans inertia | {kmeans.inertia_:,.1f} |",
        "",
        "## Overview Plots",
        "",
        f"![K=6 group sizes]({rel_size})",
        "",
        f"![K=6 PCA groups]({rel_pca})",
        "",
        f"![1D latent histograms by group]({rel_1d})",
        "",
        "## Group Sizes",
        "",
        "| group | count | fraction |",
        "|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(f"| {row['group']} | {int(row['count']):,} | {float(row['fraction']):.2%} |")
    lines += [
        "",
        "## Per-Group Pairwise Histogram Sheets",
        "",
    ]
    for group in range(args.k):
        img = group_dir / f"group_{group:02d}_all_pair_histograms.png"
        rel_img = img.relative_to(args.out.parent).as_posix()
        lines += [
            f"### Group {group}",
            "",
            f"- particles: `{counts[group]:,}`",
            "",
            f"![group {group} pair histograms]({rel_img})",
            "",
        ]
    lines += [
        "## Local Artifacts",
        "",
        f"- labels and centers: `{(args.local_out / f'kmeans_k{args.k}_labels.npz').as_posix()}`",
        f"- group summary: `{(args.local_out / f'kmeans_k{args.k}_group_summary.csv').as_posix()}`",
        f"- pair histogram stats: `{(args.local_out / f'kmeans_k{args.k}_pair_histogram_stats.csv').as_posix()}`",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

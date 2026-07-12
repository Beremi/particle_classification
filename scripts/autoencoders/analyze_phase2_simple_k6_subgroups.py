#!/usr/bin/env python3
"""Estimate latent subgroups inside the coarse K=6 simple z8 groups."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_indices(indices: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if indices.size <= limit:
        return indices
    return rng.choice(indices, size=limit, replace=False)


def best_elbow_k(ks: list[int], inertias: list[float]) -> int:
    """Pick an elbow from normalized distance to the line between endpoints."""
    if len(ks) <= 2:
        return ks[-1]
    x = np.asarray(ks, dtype=np.float64)
    y = np.asarray(inertias, dtype=np.float64)
    x = (x - x.min()) / max(1e-12, x.max() - x.min())
    y = (y - y.min()) / max(1e-12, y.max() - y.min())
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line = p2 - p1
    denom = np.linalg.norm(line)
    if denom <= 1e-12:
        return ks[0]
    pts = np.stack([x, y], axis=1)
    rel = pts - p1
    dist = np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / denom
    return int(ks[int(np.argmax(dist))])


def plot_sweep(group: int, rows: list[dict[str, object]], out: Path) -> None:
    ks = np.asarray([int(r["sub_k"]) for r in rows if int(r["sub_k"]) >= 2])
    sil = np.asarray([float(r["silhouette"]) for r in rows if int(r["sub_k"]) >= 2])
    db = np.asarray([float(r["davies_bouldin"]) for r in rows if int(r["sub_k"]) >= 2])
    bic_rows = [r for r in rows if int(r["sub_k"]) >= 1]
    bic_ks = np.asarray([int(r["sub_k"]) for r in bic_rows])
    bic = np.asarray([float(r["gmm_diag_bic"]) for r in bic_rows])
    inertia = np.asarray([float(r["inertia_per_particle"]) for r in bic_rows])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=160)
    axes[0, 0].plot(bic_ks, inertia, marker="o")
    axes[0, 0].set_title("KMeans inertia per particle")
    axes[0, 0].set_xlabel("sub-K")
    axes[0, 0].set_ylabel("inertia / n")
    axes[0, 1].plot(ks, sil, marker="o", color="#228833")
    axes[0, 1].set_title("Silhouette, higher is better")
    axes[0, 1].set_xlabel("sub-K")
    axes[0, 1].set_ylabel("silhouette")
    axes[1, 0].plot(ks, db, marker="o", color="#cc6677")
    axes[1, 0].set_title("Davies-Bouldin, lower is better")
    axes[1, 0].set_xlabel("sub-K")
    axes[1, 0].set_ylabel("DB")
    axes[1, 1].plot(bic_ks, bic, marker="o", color="#aa4499")
    axes[1, 1].set_title("Diagonal GMM BIC, lower is better")
    axes[1, 1].set_xlabel("components")
    axes[1, 1].set_ylabel("BIC")
    fig.suptitle(f"Coarse group {group}: subgroup diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out)
    plt.close(fig)


def plot_group_pca(
    group: int,
    z: np.ndarray,
    labels: np.ndarray,
    chosen_k: int,
    seed: int,
    out: Path,
) -> None:
    pca = PCA(n_components=2, random_state=seed)
    pc = pca.fit_transform(z)
    km = MiniBatchKMeans(n_clusters=chosen_k, batch_size=8192, n_init=8, random_state=seed)
    sublabels = km.fit_predict(z)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=170)
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=sublabels, s=2.0, cmap="tab20", alpha=0.65, linewidths=0)
    ax.set_title(f"Coarse group {group}: PCA view with KMeans sub-K={chosen_k}")
    ax.set_xlabel("PC1 inside group")
    ax.set_ylabel("PC2 inside group")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("subgroup")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-npz",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_latent_groups_v001/latents_and_groups_k32.npz"),
    )
    parser.add_argument(
        "--k6-labels",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_k6_group_histograms_v001/kmeans_k6_labels.npz"),
    )
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/latent_analysis/phase2-simple-z8-k6-subgroup-diagnostics.md"))
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("experimental_notes/assets/phase2_simple_z8_k6_subgroup_diagnostics_v001"),
    )
    parser.add_argument(
        "--local-out",
        type=Path,
        default=Path("local_data/experiments/phase2_simple_z8_k6_subgroup_diagnostics_v001"),
    )
    parser.add_argument("--max-sub-k", type=int, default=12)
    parser.add_argument("--sample-per-group", type=int, default=80000)
    parser.add_argument("--metric-sample", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    ensure_dir(args.asset_dir)
    ensure_dir(args.local_out)
    ensure_dir(args.out.parent)

    data = np.load(args.latent_npz)
    labels_data = np.load(args.k6_labels)
    z_norm = np.asarray(data["z_norm"], dtype=np.float32)
    labels = np.asarray(labels_data["labels"], dtype=np.int32)
    if z_norm.shape[0] != labels.shape[0]:
        raise ValueError(f"latent/label length mismatch: {z_norm.shape[0]} vs {labels.shape[0]}")

    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    groups = sorted(int(g) for g in np.unique(labels))

    for group in groups:
        group_indices = np.flatnonzero(labels == group)
        idx = sample_indices(group_indices, args.sample_per_group, rng)
        z_group = z_norm[idx]
        metric_idx = np.arange(z_group.shape[0])
        if metric_idx.size > args.metric_sample:
            metric_idx = rng.choice(metric_idx, size=args.metric_sample, replace=False)
        z_metric = z_group[metric_idx]

        group_rows: list[dict[str, object]] = []
        inertias: list[float] = []
        ks_for_elbow: list[int] = []
        for sub_k in range(1, args.max_sub_k + 1):
            km = MiniBatchKMeans(n_clusters=sub_k, batch_size=8192, n_init=8, random_state=args.seed + group * 100 + sub_k)
            sublabels = km.fit_predict(z_group)
            inertia_per_particle = float(km.inertia_ / z_group.shape[0])
            ks_for_elbow.append(sub_k)
            inertias.append(inertia_per_particle)

            metric_labels = sublabels[metric_idx]
            if sub_k >= 2 and len(np.unique(metric_labels)) > 1:
                sil = float(silhouette_score(z_metric, metric_labels))
                db = float(davies_bouldin_score(z_metric, metric_labels))
                ch = float(calinski_harabasz_score(z_metric, metric_labels))
            else:
                sil = float("nan")
                db = float("nan")
                ch = float("nan")

            gmm = GaussianMixture(
                n_components=sub_k,
                covariance_type="diag",
                max_iter=200,
                reg_covar=1e-5,
                random_state=args.seed + group * 1000 + sub_k,
            )
            gmm.fit(z_metric)
            bic = float(gmm.bic(z_metric))
            aic = float(gmm.aic(z_metric))

            row = {
                "group": group,
                "group_count": int(group_indices.size),
                "sample_count": int(z_group.shape[0]),
                "metric_sample_count": int(z_metric.shape[0]),
                "sub_k": sub_k,
                "inertia_per_particle": inertia_per_particle,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "gmm_diag_bic": bic,
                "gmm_diag_aic": aic,
            }
            group_rows.append(row)
            all_rows.append(row)

        valid_sil = [r for r in group_rows if int(r["sub_k"]) >= 2 and np.isfinite(float(r["silhouette"]))]
        valid_db = [r for r in group_rows if int(r["sub_k"]) >= 2 and np.isfinite(float(r["davies_bouldin"]))]
        best_sil = max(valid_sil, key=lambda r: float(r["silhouette"]))
        best_db = min(valid_db, key=lambda r: float(r["davies_bouldin"]))
        best_bic = min(group_rows, key=lambda r: float(r["gmm_diag_bic"]))
        elbow = best_elbow_k(ks_for_elbow, inertias)

        chosen = int(best_bic["sub_k"])
        if chosen == args.max_sub_k:
            # BIC often keeps improving on broad manifolds; use silhouette/elbow as a conservative visual estimate.
            chosen = int(best_sil["sub_k"])

        summary_rows.append(
            {
                "group": group,
                "group_count": int(group_indices.size),
                "best_silhouette_k": int(best_sil["sub_k"]),
                "best_silhouette": float(best_sil["silhouette"]),
                "best_db_k": int(best_db["sub_k"]),
                "best_db": float(best_db["davies_bouldin"]),
                "best_bic_k": int(best_bic["sub_k"]),
                "best_bic": float(best_bic["gmm_diag_bic"]),
                "elbow_k": int(elbow),
                "chosen_inspection_k": int(chosen),
            }
        )

        plot_sweep(group, group_rows, args.asset_dir / f"group_{group:02d}_subk_sweep.png")
        plot_group_pca(group, z_group if z_group.shape[0] <= args.metric_sample else z_metric, labels, chosen, args.seed, args.asset_dir / f"group_{group:02d}_chosen_subgroups.png")

    sweep_csv = args.local_out / "subgroup_sweep_metrics.csv"
    summary_csv = args.local_out / "subgroup_summary.csv"
    write_csv(
        sweep_csv,
        all_rows,
        [
            "group",
            "group_count",
            "sample_count",
            "metric_sample_count",
            "sub_k",
            "inertia_per_particle",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "gmm_diag_bic",
            "gmm_diag_aic",
        ],
    )
    write_csv(
        summary_csv,
        summary_rows,
        [
            "group",
            "group_count",
            "best_silhouette_k",
            "best_silhouette",
            "best_db_k",
            "best_db",
            "best_bic_k",
            "best_bic",
            "elbow_k",
            "chosen_inspection_k",
        ],
    )

    lines = [
        "# Simple z8 K=6 Subgroup Diagnostics",
        "",
        "This numerically checks whether each coarse K=6 latent group is internally multi-modal. "
        "Each group is sampled, then independently swept over sub-K values using KMeans and diagonal Gaussian mixtures.",
        "",
        "## How To Read This",
        "",
        "- `best_silhouette_k`: compact/separated KMeans estimate. Higher silhouette is better.",
        "- `best_db_k`: Davies-Bouldin estimate. Lower is better.",
        "- `best_bic_k`: diagonal GMM component count. Lower BIC is better, but it may keep increasing K on broad continuous manifolds.",
        "- `elbow_k`: conservative inertia elbow estimate.",
        "- `chosen_inspection_k`: a practical display estimate, using BIC unless it hits the sweep limit, then falling back to silhouette.",
        "",
        "## Summary",
        "",
        "| group | particles | best silhouette K | silhouette | best DB K | DB | best BIC K | elbow K | chosen inspection K |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['group']} | {int(row['group_count']):,} | {row['best_silhouette_k']} | "
            f"{float(row['best_silhouette']):.4f} | {row['best_db_k']} | {float(row['best_db']):.4f} | "
            f"{row['best_bic_k']} | {row['elbow_k']} | {row['chosen_inspection_k']} |"
        )
    lines += [
        "",
        "## Per-Group Diagnostics",
        "",
    ]
    for row in summary_rows:
        group = int(row["group"])
        rel_sweep = (args.asset_dir / f"group_{group:02d}_subk_sweep.png").relative_to(args.out.parent).as_posix()
        rel_pca = (args.asset_dir / f"group_{group:02d}_chosen_subgroups.png").relative_to(args.out.parent).as_posix()
        lines += [
            f"### Group {group}",
            "",
            f"- particles: `{int(row['group_count']):,}`",
            f"- chosen inspection K: `{int(row['chosen_inspection_k'])}`",
            "",
            f"![group {group} sweep]({rel_sweep})",
            "",
            f"![group {group} chosen subgroups]({rel_pca})",
            "",
        ]
    lines += [
        "## Local Artifacts",
        "",
        f"- sweep metrics: `{sweep_csv.as_posix()}`",
        f"- summary: `{summary_csv.as_posix()}`",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from particle_classification.experiments.autoencoders.canonical_voxel import _render_rows
from particle_classification.experiments.autoencoders.voxel import (
    VoxelGridConfig,
    VoxelMLPConfig,
    VoxelPatchMLPAutoencoder,
    VoxelSparseCache,
)


def source_folder(path: str) -> str:
    parts = Path(path).parts
    return parts[0] if parts else path


def hit_bucket(n_hits: int) -> str:
    if n_hits <= 4:
        return "2-4"
    if n_hits <= 10:
        return "5-10"
    if n_hits <= 50:
        return "11-50"
    return "51+"


def load_model(checkpoint: Path, device: str) -> tuple[VoxelPatchMLPAutoencoder, VoxelGridConfig, dict[str, object]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = VoxelMLPConfig(**dict(payload["model_config"]))
    grid_config = VoxelGridConfig(**dict(payload["grid_config"]))
    model = VoxelPatchMLPAutoencoder(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, grid_config, payload


def encode_split(
    *,
    model: VoxelPatchMLPAutoencoder,
    grid: VoxelGridConfig,
    cache: Path,
    split: str,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    dataset = VoxelSparseCache(cache, split=split, grid_config=grid, seed=seed)
    z_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(dataset.rows), batch_size):
            rows = dataset.rows[start : start + batch_size]
            dense = dataset.rows_to_dense(rows, device=device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                z_shape, _ = model.encode(dense)
            z_batches.append(z_shape.float().detach().cpu().numpy().astype(np.float32))
            if start and start % (batch_size * 25) == 0:
                print(f"{split}: encoded {start:,}/{len(dataset.rows):,}", flush=True)
    for row in dataset.rows:
        row["split"] = split
    return np.vstack(z_batches).astype(np.float32), dataset.rows


def pca_scores(z: np.ndarray) -> dict[str, np.ndarray]:
    mean = z.mean(axis=0)
    centered = z - mean
    cov = centered.T @ centered / max(1, len(z) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    explained = evals / max(float(evals.sum()), 1e-12)
    return {
        "mean": mean.astype(np.float32),
        "components": evecs.T.astype(np.float32),
        "explained": explained.astype(np.float32),
        "scores": (centered @ evecs).astype(np.float32),
    }


def metadata_arrays(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        "n_hits": np.asarray([int(float(row.get("n_hits", 0))) for row in rows], dtype=np.float32),
        "occupied_voxels": np.asarray([int(float(row.get("occupied_voxels", 0))) for row in rows], dtype=np.float32),
        "x_span": np.asarray([float(row.get("x_span", 0.0)) for row in rows], dtype=np.float32),
        "y_span": np.asarray([float(row.get("y_span", 0.0)) for row in rows], dtype=np.float32),
        "time_span": np.asarray([float(row.get("time_span", 0.0)) for row in rows], dtype=np.float32),
        "energy": np.asarray([float(row.get("kept_energy_fraction", 1.0)) for row in rows], dtype=np.float32),
    }


def zscore(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std = np.maximum(std, 1e-6)
    return ((z - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def run_k_sweep(
    z_norm: np.ndarray,
    *,
    ks: list[int],
    seed: int,
    sample_size: int,
    batch_size: int,
) -> tuple[list[dict[str, float]], dict[int, np.ndarray], dict[int, MiniBatchKMeans]]:
    rng = np.random.default_rng(seed)
    sample_count = min(sample_size, len(z_norm))
    sample_idx = rng.choice(len(z_norm), size=sample_count, replace=False)
    labels_by_k: dict[int, np.ndarray] = {}
    models: dict[int, MiniBatchKMeans] = {}
    rows: list[dict[str, float]] = []
    for k in ks:
        print(f"fitting MiniBatchKMeans k={k}", flush=True)
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=batch_size,
            n_init=5,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(z_norm)
        labels_by_k[k] = labels.astype(np.int32)
        models[k] = model
        sample_labels = labels[sample_idx]
        sample_z = z_norm[sample_idx]
        unique = np.unique(sample_labels)
        if len(unique) > 1:
            sil = float(silhouette_score(sample_z, sample_labels, metric="euclidean"))
            db = float(davies_bouldin_score(sample_z, sample_labels))
            ch = float(calinski_harabasz_score(sample_z, sample_labels))
        else:
            sil = db = ch = float("nan")
        counts = np.bincount(labels, minlength=k)
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette_sample": sil,
                "davies_bouldin_sample": db,
                "calinski_harabasz_sample": ch,
                "min_cluster": int(counts.min()),
                "max_cluster": int(counts.max()),
                "median_cluster": float(np.median(counts)),
            }
        )
    return rows, labels_by_k, models


def cluster_summary(
    *,
    labels: np.ndarray,
    z_norm: np.ndarray,
    rows: list[dict[str, str]],
    meta: dict[str, np.ndarray],
    centers: np.ndarray,
) -> tuple[list[dict[str, object]], list[int]]:
    summaries: list[dict[str, object]] = []
    medoids: list[int] = []
    for cluster in range(centers.shape[0]):
        idx = np.flatnonzero(labels == cluster)
        if idx.size == 0:
            continue
        dist = np.sum((z_norm[idx] - centers[cluster]) ** 2, axis=1)
        medoid = int(idx[int(np.argmin(dist))])
        medoids.append(medoid)
        buckets = Counter(hit_bucket(int(meta["n_hits"][i])) for i in idx)
        folders = Counter(source_folder(rows[i]["source_path"]) for i in idx)
        summaries.append(
            {
                "cluster": int(cluster),
                "count": int(idx.size),
                "fraction": float(idx.size / len(labels)),
                "medoid_global_index": medoid,
                "median_hits": float(np.median(meta["n_hits"][idx])),
                "p90_hits": float(np.quantile(meta["n_hits"][idx], 0.90)),
                "median_voxels": float(np.median(meta["occupied_voxels"][idx])),
                "median_x_span": float(np.median(meta["x_span"][idx])),
                "median_y_span": float(np.median(meta["y_span"][idx])),
                "median_time_span": float(np.median(meta["time_span"][idx])),
                "dominant_hit_bucket": buckets.most_common(1)[0][0],
                "top_source_folder": folders.most_common(1)[0][0],
            }
        )
    summaries.sort(key=lambda item: int(item["count"]), reverse=True)
    return summaries, medoids


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_cluster_gallery(
    *,
    summaries: list[dict[str, object]],
    rows: list[dict[str, str]],
    cache: Path,
    grid: VoxelGridConfig,
    asset_dir: Path,
    report_parent: Path,
    device: str,
    max_clusters: int,
) -> list[dict[str, object]]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, VoxelSparseCache] = {}
    rendered: list[dict[str, object]] = []
    for rank, summary in enumerate(summaries[:max_clusters], start=1):
        idx = int(summary["medoid_global_index"])
        row = rows[idx]
        split = str(row["split"])
        if split not in datasets:
            datasets[split] = VoxelSparseCache(cache, split=split, grid_config=grid)
        dense = datasets[split].rows_to_dense([row], device=device)
        target = dense[0].float().detach().cpu().numpy()
        image = asset_dir / f"cluster_{int(summary['cluster']):02d}_rank_{rank:02d}.png"
        _render_rows(
            [(f"cluster {summary['cluster']} medoid", target)],
            image,
            f"cluster {summary['cluster']} medoid hits {row['n_hits']} source {Path(row['source_path']).name}",
        )
        rendered.append(
            {
                "rank": rank,
                "cluster": int(summary["cluster"]),
                "global_index": idx,
                "source_path": row["source_path"],
                "particle_id": row["particle_id"],
                "n_hits": row["n_hits"],
                "image": image.relative_to(report_parent).as_posix(),
            }
        )
    return rendered


def make_plots(
    *,
    pca: dict[str, np.ndarray],
    labels: np.ndarray,
    summaries: list[dict[str, object]],
    scores_rows: list[dict[str, float]],
    meta: dict[str, np.ndarray],
    asset_dir: Path,
    report_parent: Path,
    seed: int,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    asset_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    sample_n = min(160_000, len(labels))
    sample = rng.choice(len(labels), size=sample_n, replace=False)
    scores = pca["scores"][sample]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(1, len(pca["explained"]) + 1)
    ax.bar(x, pca["explained"] * 100.0)
    ax.plot(x, np.cumsum(pca["explained"]) * 100.0, marker="o", color="crimson")
    ax.set_xlabel("latent PCA component")
    ax.set_ylabel("variance explained [%]")
    ax.set_title("z8 latent PCA spectrum")
    path = asset_dir / "latent_pca_spectrum.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["pca_spectrum"] = path.relative_to(report_parent).as_posix()

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=labels[sample], s=2, alpha=0.35, cmap="tab20", linewidths=0)
    ax.set_xlabel(f"PC1 ({pca['explained'][0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca['explained'][1] * 100:.1f}%)")
    ax.set_title("Latent-space groups on PC1/PC2")
    fig.colorbar(sc, ax=ax, label="k-means group")
    path = asset_dir / "pc1_pc2_clusters.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["pc12_clusters"] = path.relative_to(report_parent).as_posix()

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=np.log1p(meta["n_hits"][sample]), s=2, alpha=0.3, cmap="viridis", linewidths=0)
    ax.set_xlabel(f"PC1 ({pca['explained'][0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca['explained'][1] * 100:.1f}%)")
    ax.set_title("Latent PC1/PC2 colored by hit count")
    fig.colorbar(sc, ax=ax, label="log(1 + hits)")
    path = asset_dir / "pc1_pc2_hits.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["pc12_hits"] = path.relative_to(report_parent).as_posix()

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    score_df = np.asarray([[row["k"], row["silhouette_sample"], row["davies_bouldin_sample"]] for row in scores_rows], dtype=float)
    ax.plot(score_df[:, 0], score_df[:, 1], marker="o", label="silhouette higher better")
    ax2 = ax.twinx()
    ax2.plot(score_df[:, 0], score_df[:, 2], marker="s", color="crimson", label="Davies-Bouldin lower better")
    ax.set_xlabel("K")
    ax.set_ylabel("silhouette")
    ax2.set_ylabel("Davies-Bouldin")
    ax.set_title("K sweep on z-scored latent vectors")
    ax.grid(alpha=0.25)
    path = asset_dir / "k_sweep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["k_sweep"] = path.relative_to(report_parent).as_posix()

    top = summaries[:32]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar([str(item["cluster"]) for item in top], [int(item["count"]) for item in top])
    ax.set_xlabel("cluster")
    ax.set_ylabel("particles")
    ax.set_title("Selected group sizes, top 32 by count")
    ax.tick_params(axis="x", rotation=90)
    path = asset_dir / "cluster_sizes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["cluster_sizes"] = path.relative_to(report_parent).as_posix()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode simple voxel-AE latents and group particles in latent space.")
    parser.add_argument("--checkpoint", type=Path, default=Path("local_data/experiments/phase2_simple_voxel_ae_z8_b1024_clean_fixedmine_20phase_v001/checkpoint.pt"))
    parser.add_argument("--cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_8x32x32_representative_v001"))
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/latent_analysis/phase2-simple-z8-latent-groups.md"))
    parser.add_argument("--asset-dir", type=Path, default=Path("experimental_notes/assets/phase2_simple_z8_latent_groups_v001"))
    parser.add_argument("--local-out", type=Path, default=Path("local_data/experiments/phase2_simple_z8_latent_groups_v001"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--cluster-batch-size", type=int, default=20000)
    parser.add_argument("--ks", default="8,16,24,32,48,64")
    parser.add_argument("--selected-k", type=int, default=32)
    parser.add_argument("--metric-sample", type=int, default=20000)
    parser.add_argument("--prototype-clusters", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    args.local_out.mkdir(parents=True, exist_ok=True)
    args.asset_dir.mkdir(parents=True, exist_ok=True)

    model, grid, checkpoint_payload = load_model(args.checkpoint, device)
    all_z: list[np.ndarray] = []
    all_rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        z, rows = encode_split(model=model, grid=grid, cache=args.cache, split=split, batch_size=args.batch_size, device=device, seed=args.seed)
        all_z.append(z)
        all_rows.extend(rows)
    z_shape = np.vstack(all_z).astype(np.float32)
    z_norm, z_mean, z_std = zscore(z_shape)
    meta = metadata_arrays(all_rows)
    pca = pca_scores(z_norm)

    ks = [int(part) for part in args.ks.split(",") if part.strip()]
    if args.selected_k not in ks:
        ks.append(args.selected_k)
        ks = sorted(set(ks))
    score_rows, labels_by_k, models = run_k_sweep(
        z_norm,
        ks=ks,
        seed=args.seed,
        sample_size=args.metric_sample,
        batch_size=args.cluster_batch_size,
    )
    labels = labels_by_k[args.selected_k]
    centers = models[args.selected_k].cluster_centers_.astype(np.float32)
    summaries, medoids = cluster_summary(labels=labels, z_norm=z_norm, rows=all_rows, meta=meta, centers=centers)

    write_csv(args.local_out / "kmeans_scores.csv", score_rows)
    write_csv(args.local_out / f"group_summary_k{args.selected_k}.csv", summaries)
    with (args.local_out / f"particle_latent_groups_k{args.selected_k}.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "global_index",
            "split",
            "cluster",
            "source_path",
            "particle_id",
            "n_hits",
            "occupied_voxels",
            "x_span",
            "y_span",
            "time_span",
            "pc1",
            "pc2",
            "pc3",
            "z0",
            "z1",
            "z2",
            "z3",
            "z4",
            "z5",
            "z6",
            "z7",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(all_rows):
            writer.writerow(
                {
                    "global_index": idx,
                    "split": row["split"],
                    "cluster": int(labels[idx]),
                    "source_path": row["source_path"],
                    "particle_id": row["particle_id"],
                    "n_hits": row["n_hits"],
                    "occupied_voxels": row.get("occupied_voxels", ""),
                    "x_span": row.get("x_span", ""),
                    "y_span": row.get("y_span", ""),
                    "time_span": row.get("time_span", ""),
                    "pc1": f"{pca['scores'][idx, 0]:.7g}",
                    "pc2": f"{pca['scores'][idx, 1]:.7g}",
                    "pc3": f"{pca['scores'][idx, 2]:.7g}",
                    **{f"z{j}": f"{z_shape[idx, j]:.7g}" for j in range(z_shape.shape[1])},
                }
            )

    np.savez_compressed(
        args.local_out / f"latents_and_groups_k{args.selected_k}.npz",
        z_shape=z_shape,
        z_norm=z_norm,
        z_mean=z_mean,
        z_std=z_std,
        labels=labels.astype(np.int32),
        pca_scores=pca["scores"],
        pca_components=pca["components"],
        pca_explained=pca["explained"],
        centers=centers,
    )

    plots = make_plots(
        pca=pca,
        labels=labels,
        summaries=summaries,
        scores_rows=score_rows,
        meta=meta,
        asset_dir=args.asset_dir,
        report_parent=args.out.parent,
        seed=args.seed,
    )
    rendered = render_cluster_gallery(
        summaries=summaries,
        rows=all_rows,
        cache=args.cache,
        grid=grid,
        asset_dir=args.asset_dir / "prototypes",
        report_parent=args.out.parent,
        device=device,
        max_clusters=args.prototype_clusters,
    )
    write_csv(args.local_out / f"rendered_prototypes_k{args.selected_k}.csv", rendered)

    bucket_by_cluster: dict[int, Counter[str]] = defaultdict(Counter)
    for idx, label in enumerate(labels):
        bucket_by_cluster[int(label)][hit_bucket(int(meta["n_hits"][idx]))] += 1

    lines = [
        "# Simple z8 Latent-Space Grouping",
        "",
        f"Checkpoint: `{args.checkpoint.as_posix()}`",
        f"Dataset cache: `{args.cache.as_posix()}`",
        "",
        "This report encodes all cached particles with the clean z8 voxel autoencoder and groups the 8D latent vectors with MiniBatchKMeans. The groups are unsupervised morphology/encoding groups, not physical particle labels.",
        "",
        "## Output Artifacts",
        "",
        f"- Full particle-to-group table: `{(args.local_out / f'particle_latent_groups_k{args.selected_k}.csv').as_posix()}`",
        f"- Latent arrays and labels: `{(args.local_out / f'latents_and_groups_k{args.selected_k}.npz').as_posix()}`",
        f"- K sweep scores: `{(args.local_out / 'kmeans_scores.csv').as_posix()}`",
        f"- Group summary: `{(args.local_out / f'group_summary_k{args.selected_k}.csv').as_posix()}`",
        "",
        "## Dataset",
        "",
        "| item | value |",
        "|---|---:|",
        f"| encoded particles | {len(all_rows):,} |",
        f"| latent dimensions | {z_shape.shape[1]} |",
        f"| selected K | {args.selected_k} |",
        f"| checkpoint best step | {checkpoint_payload.get('best_step', '')} |",
        f"| checkpoint best val L2 | {float(checkpoint_payload.get('best_val_loss', float('nan'))):.6f} |",
        "",
        "## PCA And Group Plots",
        "",
        f"![latent PCA spectrum]({plots['pca_spectrum']})",
        "",
        f"![PC1/PC2 clusters]({plots['pc12_clusters']})",
        "",
        f"![PC1/PC2 hit count]({plots['pc12_hits']})",
        "",
        f"![K sweep]({plots['k_sweep']})",
        "",
        f"![cluster sizes]({plots['cluster_sizes']})",
        "",
        "## K Sweep",
        "",
        "| K | silhouette sample | Davies-Bouldin sample | Calinski-Harabasz sample | min cluster | max cluster | median cluster |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in score_rows:
        lines.append(
            f"| {int(row['k'])} | {row['silhouette_sample']:.4f} | {row['davies_bouldin_sample']:.4f} | "
            f"{row['calinski_harabasz_sample']:.1f} | {int(row['min_cluster'])} | {int(row['max_cluster'])} | {row['median_cluster']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"## Selected Groups: K={args.selected_k}",
            "",
            "| rank | cluster | count | fraction | median hits | p90 hits | median voxels | median spans x/y/t | dominant bucket | top source folder | medoid |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    rendered_by_cluster = {int(item["cluster"]): item for item in rendered}
    for rank, item in enumerate(summaries[:32], start=1):
        image = rendered_by_cluster.get(int(item["cluster"]), {}).get("image")
        medoid = f"[image]({image})" if image else str(item["medoid_global_index"])
        lines.append(
            f"| {rank} | {item['cluster']} | {int(item['count']):,} | {item['fraction'] * 100:.2f}% | "
            f"{item['median_hits']:.1f} | {item['p90_hits']:.1f} | {item['median_voxels']:.1f} | "
            f"{item['median_x_span']:.1f}/{item['median_y_span']:.1f}/{item['median_time_span']:.2f} | "
            f"{item['dominant_hit_bucket']} | `{item['top_source_folder']}` | {medoid} |"
        )
    lines.extend(["", "## Prototype Gallery", ""])
    for item in rendered:
        lines.extend(
            [
                f"### Cluster {item['cluster']} (rank {item['rank']})",
                "",
                f"- source: `{item['source_path']}`",
                f"- particle: `{item['particle_id']}`",
                f"- hits: `{item['n_hits']}`",
                "",
                f"![cluster {item['cluster']}]({item['image']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- The clustering is over z-scored `z_shape` only.",
            "- KMeans imposes groups even if the latent manifold is continuous; use the K sweep and prototype gallery as diagnostics, not as proof of physical species.",
            "- Because this z8 autoencoder was trained for reconstruction, latent groups may still encode nuisance pose, energy, or reconstruction difficulty.",
        ]
    )
    args.out.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "checkpoint": args.checkpoint.as_posix(),
        "cache": args.cache.as_posix(),
        "particles": len(all_rows),
        "selected_k": args.selected_k,
        "k_sweep": score_rows,
        "top_groups": summaries[:32],
        "plots": plots,
        "prototype_count": len(rendered),
        "model_config": checkpoint_payload.get("model_config"),
        "grid_config": asdict(grid),
    }
    (args.local_out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

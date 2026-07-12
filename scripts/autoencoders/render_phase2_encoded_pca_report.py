from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from particle_classification.experiments.autoencoders.canonical_voxel import (
    _load_canonical_checkpoint,
    _render_rows,
    volume_moments,
)
from particle_classification.experiments.autoencoders.voxel import VoxelSparseCache


def angle_diff_mod_pi(a: float, b: float) -> float:
    return abs(((a - b + math.pi / 2.0) % math.pi) - math.pi / 2.0)


def hit_bucket(n_hits: int) -> str:
    if n_hits <= 4:
        return "2-4"
    if n_hits <= 10:
        return "5-10"
    if n_hits <= 50:
        return "11-50"
    return "51+"


def source_folder(path: str) -> str:
    return Path(path).parts[0] if Path(path).parts else path


def encode_split(
    *,
    checkpoint: Path,
    cache: Path,
    split: str,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], object, object]:
    model, grid_config, checkpoint_payload = _load_canonical_checkpoint(checkpoint, device)
    dataset = VoxelSparseCache(cache, split=split, grid_config=grid_config, seed=seed)
    z_batches: list[np.ndarray] = []
    transform_batches: list[np.ndarray] = []
    energy_batches: list[np.ndarray] = []
    theta_batches: list[np.ndarray] = []
    theta_time_batches: list[np.ndarray] = []
    scale_batches: list[np.ndarray] = []
    for start in range(0, len(dataset.rows), batch_size):
        rows = dataset.rows[start : start + batch_size]
        dense = dataset.rows_to_dense(rows, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            z_shape, z_transform = model.encode(dense)
        with torch.no_grad():
            moments = volume_moments(dense.float(), model.config)
        z_batches.append(z_shape.float().detach().cpu().numpy())
        transform_batches.append(z_transform.float().detach().cpu().numpy())
        energy_batches.append(moments["energy"].float().detach().cpu().numpy())
        theta_batches.append(moments["target_transform"][:, 3].float().detach().cpu().numpy())
        theta_time_batches.append(moments["target_transform"][:, 4].float().detach().cpu().numpy())
        scale_batches.append(moments["target_transform"][:, 5].float().detach().cpu().numpy())
        if start and start % (batch_size * 25) == 0:
            print(f"{split}: encoded {start:,}/{len(dataset.rows):,}", flush=True)
    arrays = {
        "z_shape": np.vstack(z_batches).astype(np.float32),
        "z_transform": np.vstack(transform_batches).astype(np.float32),
        "energy": np.concatenate(energy_batches).astype(np.float32),
        "theta_xy_pca": np.concatenate(theta_batches).astype(np.float32),
        "theta_time_pca": np.concatenate(theta_time_batches).astype(np.float32),
        "scale_pca": np.concatenate(scale_batches).astype(np.float32),
    }
    return arrays, dataset.rows, model, checkpoint_payload


def pca_from_latent(z: np.ndarray) -> dict[str, np.ndarray]:
    mean = z.mean(axis=0)
    centered = z - mean
    cov = centered.T @ centered / max(1, z.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    explained = evals / np.maximum(evals.sum(), 1e-12)
    scores = centered @ evecs
    return {
        "mean": mean.astype(np.float32),
        "components": evecs.T.astype(np.float32),
        "eigenvalues": evals.astype(np.float32),
        "explained": explained.astype(np.float32),
        "scores": scores.astype(np.float32),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def describe_pc(correlations: dict[str, float]) -> str:
    ranked = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
    return ", ".join(f"{name} {value:+.2f}" for name, value in ranked[:4])


def representative_indices(scores: np.ndarray, pc: int) -> tuple[int, int, int]:
    values = scores[:, pc]
    lo_target = np.quantile(values, 0.005)
    hi_target = np.quantile(values, 0.995)
    mid_target = np.median(values)
    lo = int(np.argmin(np.abs(values - lo_target)))
    mid = int(np.argmin(np.abs(values - mid_target)))
    hi = int(np.argmin(np.abs(values - hi_target)))
    return lo, mid, hi


def render_pc_examples(
    *,
    pc: int,
    indices: tuple[int, int, int],
    rows: list[dict[str, str]],
    split_offsets: dict[str, tuple[int, int]],
    cache: Path,
    checkpoint: Path,
    asset_dir: Path,
    out_parent: Path,
    device: str,
) -> str:
    model, grid_config, _ = _load_canonical_checkpoint(checkpoint, device)
    volumes: list[tuple[str, np.ndarray]] = []
    labels = ["negative tail", "center", "positive tail"]
    for label, global_index in zip(labels, indices, strict=True):
        row = rows[global_index]
        split = str(row["split"])
        dataset = VoxelSparseCache(cache, split=split, grid_config=grid_config)
        dense = dataset.rows_to_dense([row], device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            recon = model(dense)["reconstruction"]
        target_np = dense[0].float().detach().cpu().numpy()
        recon_np = recon[0].float().detach().cpu().numpy()
        volumes.append((f"{label} target", target_np))
        volumes.append((f"{label} recon", recon_np))
    image = asset_dir / f"pc{pc + 1:02d}_extremes.png"
    _render_rows(volumes, image, f"latent PCA component {pc + 1} extremes")
    return image.relative_to(out_parent).as_posix()


def make_plots(
    *,
    pca: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    asset_dir: Path,
    out_parent: Path,
    seed: int,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}
    explained = pca["explained"]
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained * 100.0, color="#4f83cc", label="per component")
    ax.plot(x, cumulative * 100.0, color="#c44536", marker="o", label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained [%]")
    ax.set_title("z_shape PCA spectrum")
    ax.set_xticks(x)
    ax.legend()
    path = asset_dir / "pca_variance_spectrum.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["variance"] = path.relative_to(out_parent).as_posix()

    rng = np.random.default_rng(seed)
    count = min(120_000, pca["scores"].shape[0])
    sample = rng.choice(pca["scores"].shape[0], size=count, replace=False)
    scores = pca["scores"][sample]
    n_hits = metadata["n_hits"][sample]
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=np.log1p(n_hits), s=2, alpha=0.25, cmap="viridis", linewidths=0)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title("Encoded particles in PC1/PC2")
    fig.colorbar(sc, ax=ax, label="log(1 + hit count)")
    path = asset_dir / "pc1_pc2_log_hits.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["pc12"] = path.relative_to(out_parent).as_posix()

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    sc = ax.scatter(scores[:, 0], scores[:, 2], c=metadata["energy"][sample], s=2, alpha=0.25, cmap="magma", linewidths=0)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC3 ({explained[2] * 100:.1f}%)")
    ax.set_title("Encoded particles in PC1/PC3")
    fig.colorbar(sc, ax=ax, label="normalized log-energy sum")
    path = asset_dir / "pc1_pc3_energy.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["pc13"] = path.relative_to(out_parent).as_posix()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA report over encoded Phase 2 particle latents.")
    parser.add_argument("--checkpoint", type=Path, default=Path("local_data/experiments/phase2_canonical_voxel_ae_compressed_plain_l2_wide_z24_depth_xyenergy_b512_v001/checkpoint.pt"))
    parser.add_argument("--cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_8x32x32_representative_v001"))
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/latent_analysis/phase2-encoded-pca-report.md"))
    parser.add_argument("--asset-dir", type=Path, default=Path("experimental_notes/assets/phase2_encoded_pca_xyenergy_v001"))
    parser.add_argument("--local-out", type=Path, default=Path("local_data/experiments/phase2_encoded_pca_xyenergy_v001"))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pcs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    args.local_out.mkdir(parents=True, exist_ok=True)

    all_arrays: list[dict[str, np.ndarray]] = []
    all_rows: list[dict[str, str]] = []
    split_offsets: dict[str, tuple[int, int]] = {}
    model_config = None
    grid_config = None
    for split in ("train", "val", "test"):
        start = len(all_rows)
        arrays, rows, _, checkpoint_payload = encode_split(
            checkpoint=args.checkpoint,
            cache=args.cache,
            split=split,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        for row in rows:
            row["split"] = split
        all_arrays.append(arrays)
        all_rows.extend(rows)
        split_offsets[split] = (start, len(all_rows))
        model_config = checkpoint_payload.get("model_config")
        grid_config = checkpoint_payload.get("grid_config")

    z = np.vstack([arrays["z_shape"] for arrays in all_arrays])
    transforms = np.vstack([arrays["z_transform"] for arrays in all_arrays])
    energy = np.concatenate([arrays["energy"] for arrays in all_arrays])
    theta_xy_pca = np.concatenate([arrays["theta_xy_pca"] for arrays in all_arrays])
    theta_time_pca = np.concatenate([arrays["theta_time_pca"] for arrays in all_arrays])
    scale_pca = np.concatenate([arrays["scale_pca"] for arrays in all_arrays])
    pca = pca_from_latent(z)
    scores = pca["scores"]

    metadata = {
        "n_hits": np.asarray([int(float(row.get("n_hits", 0))) for row in all_rows], dtype=np.float32),
        "occupied_voxels": np.asarray([int(float(row.get("occupied_voxels", 0))) for row in all_rows], dtype=np.float32),
        "x_span": np.asarray([float(row.get("x_span", 0.0)) for row in all_rows], dtype=np.float32),
        "y_span": np.asarray([float(row.get("y_span", 0.0)) for row in all_rows], dtype=np.float32),
        "time_span": np.asarray([float(row.get("time_span", 0.0)) for row in all_rows], dtype=np.float32),
        "energy": energy,
        "theta_xy_sin2": np.sin(2.0 * theta_xy_pca),
        "theta_xy_cos2": np.cos(2.0 * theta_xy_pca),
        "theta_time_pca": theta_time_pca,
        "scale_pca": scale_pca,
        "model_energy_scale": transforms[:, 6],
        "model_theta_xy": transforms[:, 3],
    }

    np.savez_compressed(
        args.local_out / "encoded_pca_arrays.npz",
        z_shape=z,
        z_transform=transforms,
        scores=scores,
        components=pca["components"],
        mean=pca["mean"],
        explained=pca["explained"],
        energy=energy,
        theta_xy_pca=theta_xy_pca,
        theta_time_pca=theta_time_pca,
        scale_pca=scale_pca,
    )

    with (args.local_out / "encoded_particle_rows.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "global_index",
            "split",
            "source_path",
            "particle_id",
            "n_hits",
            "occupied_voxels",
            "x_span",
            "y_span",
            "time_span",
            "energy",
            "pc1",
            "pc2",
            "pc3",
            "pc4",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(all_rows):
            writer.writerow(
                {
                    "global_index": idx,
                    "split": row["split"],
                    "source_path": row["source_path"],
                    "particle_id": row["particle_id"],
                    "n_hits": row["n_hits"],
                    "occupied_voxels": row.get("occupied_voxels", ""),
                    "x_span": row.get("x_span", ""),
                    "y_span": row.get("y_span", ""),
                    "time_span": row.get("time_span", ""),
                    "energy": f"{energy[idx]:.6g}",
                    "pc1": f"{scores[idx, 0]:.6g}",
                    "pc2": f"{scores[idx, 1]:.6g}",
                    "pc3": f"{scores[idx, 2]:.6g}",
                    "pc4": f"{scores[idx, 3]:.6g}",
                }
            )

    plots = make_plots(pca=pca, metadata=metadata, asset_dir=args.asset_dir, out_parent=args.out.parent, seed=args.seed)

    pc_records = []
    for pc in range(min(args.pcs, scores.shape[1])):
        indices = representative_indices(scores, pc)
        image_rel = render_pc_examples(
            pc=pc,
            indices=indices,
            rows=all_rows,
            split_offsets=split_offsets,
            cache=args.cache,
            checkpoint=args.checkpoint,
            asset_dir=args.asset_dir,
            out_parent=args.out.parent,
            device=device,
        )
        corrs = {name: corr(scores[:, pc], values) for name, values in metadata.items()}
        pc_records.append((pc, indices, image_rel, corrs))

    explained = pca["explained"]
    cumulative = np.cumsum(explained)
    threshold_counts = {
        "50%": int(np.searchsorted(cumulative, 0.50) + 1),
        "80%": int(np.searchsorted(cumulative, 0.80) + 1),
        "90%": int(np.searchsorted(cumulative, 0.90) + 1),
        "95%": int(np.searchsorted(cumulative, 0.95) + 1),
        "99%": int(np.searchsorted(cumulative, 0.99) + 1),
    }
    entropy = -float(np.sum(explained * np.log(np.maximum(explained, 1e-12))))
    effective_dim = float(np.exp(entropy))
    buckets = Counter(hit_bucket(int(float(row.get("n_hits", 0)))) for row in all_rows)
    sources = Counter(source_folder(row["source_path"]) for row in all_rows)

    lines = [
        "# Phase 2 Encoded-Latent PCA Report",
        "",
        f"Checkpoint: `{args.checkpoint.as_posix()}`",
        f"Dataset cache: `{args.cache.as_posix()}`",
        "",
        "This report encodes every cached particle view with the current model and runs PCA on `z_shape` only. The explicit transform tail is not included in the PCA, but it is used for diagnostics.",
        "",
        "## Dataset",
        "",
        "| item | value |",
        "|---|---:|",
        f"| particles encoded | {len(all_rows):,} |",
        f"| latent dimensions | {z.shape[1]} |",
        f"| transform mode | `{model_config.get('transform_mode') if isinstance(model_config, dict) else 'unknown'}` |",
        f"| grid | `{grid_config.get('t_bins')} x {grid_config.get('y_bins')} x {grid_config.get('x_bins')}` |" if isinstance(grid_config, dict) else "| grid | unknown |",
        f"| train / val / test | {split_offsets['train'][1] - split_offsets['train'][0]:,} / {split_offsets['val'][1] - split_offsets['val'][0]:,} / {split_offsets['test'][1] - split_offsets['test'][0]:,} |",
        "",
        "Hit-count buckets:",
        "",
        "| bucket | count |",
        "|---|---:|",
    ]
    for bucket in ("2-4", "5-10", "11-50", "51+"):
        lines.append(f"| {bucket} | {buckets.get(bucket, 0):,} |")
    lines.extend(["", "Largest source folders:", "", "| folder | count |", "|---|---:|"])
    for folder, count in sources.most_common(8):
        lines.append(f"| `{folder}` | {count:,} |")

    lines.extend(
        [
            "",
            "## PCA Spectrum",
            "",
            f"Effective PCA dimension from entropy of the spectrum: `{effective_dim:.2f}`.",
            "",
            "| cumulative variance | components needed |",
            "|---|---:|",
        ]
    )
    for label, count in threshold_counts.items():
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            f"![PCA variance spectrum]({plots['variance']})",
            "",
            f"![PC1 PC2 log hits]({plots['pc12']})",
            "",
            f"![PC1 PC3 energy]({plots['pc13']})",
            "",
            "## Principal Modes",
            "",
            "These are continuous PCA axes, not discrete physical labels. The interpretation column lists the strongest scalar correlations with simple particle descriptors. The images show negative-tail, central, and positive-tail examples for each component.",
            "",
            "| PC | variance | cumulative | strongest descriptor correlations | negative / center / positive examples | image |",
            "|---:|---:|---:|---|---|---|",
        ]
    )
    for pc, indices, image_rel, corrs in pc_records:
        example_text = []
        for name, idx in zip(("neg", "mid", "pos"), indices, strict=True):
            row = all_rows[idx]
            example_text.append(
                f"{name}: `{Path(row['source_path']).name}` p{row['particle_id']} h{row['n_hits']}"
            )
        lines.append(
            f"| PC{pc + 1} | {explained[pc] * 100:.2f}% | {cumulative[pc] * 100:.2f}% | "
            f"{describe_pc(corrs)} | {'; '.join(example_text)} | [png]({image_rel}) |"
        )

    for pc, indices, image_rel, corrs in pc_records:
        lines.extend(["", f"### PC{pc + 1}", "", f"![PC{pc + 1} examples]({image_rel})", ""])
        lines.append(f"- variance: `{explained[pc] * 100:.2f}%`; cumulative: `{cumulative[pc] * 100:.2f}%`")
        lines.append(f"- strongest correlations: {describe_pc(corrs)}")
        for name, idx in zip(("negative tail", "center", "positive tail"), indices, strict=True):
            row = all_rows[idx]
            lines.append(
                f"- {name}: split `{row['split']}`, source `{row['source_path']}`, particle `{row['particle_id']}`, "
                f"hits `{row['n_hits']}`, occupied voxels `{row.get('occupied_voxels', '')}`, "
                f"energy `{energy[idx]:.3f}`, PC score `{scores[idx, pc]:.3f}`"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The PCA modes describe the model's learned `z_shape`, not ground-truth particle species.",
            "- Because the `xy_energy` transform head collapsed in the previous check, orientation can still leak into `z_shape`; correlations with `theta_xy_sin2` and `theta_xy_cos2` are included to expose that.",
            f"- Full numeric arrays are saved locally at `{(args.local_out / 'encoded_pca_arrays.npz').as_posix()}`.",
            f"- Per-particle PC scores and metadata are saved locally at `{(args.local_out / 'encoded_particle_rows.csv').as_posix()}`.",
        ]
    )
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

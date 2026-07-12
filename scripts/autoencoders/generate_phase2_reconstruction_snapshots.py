#!/usr/bin/env python3
"""Generate Phase 2 original/encoded/decoded snapshots for large particles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from particle_classification.experiments.autoencoders.point import Phase2ModelConfig, Phase2ParticleModel


POINT_AXES = (0, 1, 2)
ENERGY_INDEX = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def normalize(values: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
    return (values - mean_arr) / std_arr


def denormalize(values: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.maximum(np.asarray(std, dtype=np.float32), 1e-6)
    return values * std_arr + mean_arr


def load_chunk_row(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(row["chunk_path"]), allow_pickle=False) as data:
        chunk_row = int(row["chunk_row"])
        start = int(data["offsets"][chunk_row])
        end = int(data["offsets"][chunk_row + 1])
        points = data["points"][start:end].astype(np.float32)
        summary = data["summary"][chunk_row].astype(np.float32)
    return points, summary


def load_top_rows(manifest: Path, top: int) -> pd.DataFrame:
    usecols = [
        "source_path",
        "source_npz",
        "particle_id",
        "view_id",
        "n_hits",
        "view_n_hits",
        "sample_fraction",
        "split",
        "chunk_path",
        "chunk_row",
        "energy_sum",
        "time_span",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    ]
    df = pd.read_csv(manifest, usecols=usecols)
    df = df[df["view_id"].astype(int) == 0].copy()
    df = df.sort_values(["n_hits", "energy_sum"], ascending=[False, False])
    return df.head(top).reset_index(drop=True)


def equalize_3d(ax: plt.Axes, original: np.ndarray, decoded: np.ndarray) -> None:
    pts = np.concatenate([original[:, POINT_AXES], decoded[:, POINT_AXES]], axis=0)
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.nanmax(maxs - mins) / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def scatter_particle(ax: plt.Axes, points: np.ndarray, *, title: str, size: float) -> None:
    color = np.clip(points[:, ENERGY_INDEX], 0.0, 1.0)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        cmap="viridis",
        s=size,
        alpha=0.86,
        linewidths=0.0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x centered")
    ax.set_ylabel("y centered")
    ax.set_zlabel("time scaled")
    ax.view_init(elev=24, azim=-58)


def chamfer_xyte(original: np.ndarray, decoded: np.ndarray) -> float:
    left = original[:, :5]
    right = decoded[:, :5]
    diff = left[:, None, :] - right[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return float(dist.min(axis=1).mean() + dist.min(axis=0).mean())


def plot_one(
    row: pd.Series,
    original: np.ndarray,
    decoded: np.ndarray,
    z: np.ndarray,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(13.5, 4.4), constrained_layout=True)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3)
    scatter_particle(ax1, original, title=f"input view ({len(original)} hits)", size=12)
    scatter_particle(ax2, decoded, title=f"decoded ({len(decoded)} points)", size=22)
    equalize_3d(ax1, original, decoded)
    equalize_3d(ax2, original, decoded)
    ax3.axhline(0.0, color="#444444", linewidth=0.8)
    ax3.bar(np.arange(len(z)), z, color="#3b82f6")
    ax3.set_title("encoded latent z", fontsize=10)
    ax3.set_xlabel("latent dim")
    ax3.set_ylabel("value")
    ax3.set_xticks(np.arange(len(z)))
    fig.suptitle(
        f"{row['source_path']}  particle {int(row['particle_id'])}  "
        f"n={int(row['n_hits'])}, sampled={int(row['view_n_hits'])}",
        fontsize=11,
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_montage(items: list[dict[str, object]], output: Path) -> None:
    rows = len(items)
    fig = plt.figure(figsize=(11.8, max(3.0, 2.55 * rows)), constrained_layout=True)
    for idx, item in enumerate(items):
        original = item["original"]
        decoded = item["decoded"]
        row = item["row"]
        ax1 = fig.add_subplot(rows, 2, idx * 2 + 1, projection="3d")
        ax2 = fig.add_subplot(rows, 2, idx * 2 + 2, projection="3d")
        scatter_particle(ax1, original, title=f"#{idx + 1} input p{int(row['particle_id'])} n={int(row['n_hits'])}", size=7)
        scatter_particle(ax2, decoded, title=f"#{idx + 1} decoded z={len(item['z'])}", size=15)
        equalize_3d(ax1, original, decoded)
        equalize_3d(ax2, original, decoded)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    normalization = json.loads((args.dataset / "normalization.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = Phase2ModelConfig(**checkpoint["model_config"])
    model = Phase2ParticleModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)
    model.eval()

    top_rows = load_top_rows(args.dataset / "manifest.csv", args.top)
    items: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for idx, row in top_rows.iterrows():
            raw_points, raw_summary = load_chunk_row(row)
            points = normalize(raw_points, normalization["point_mean"], normalization["point_std"])
            summary = normalize(raw_summary[None, :], normalization["summary_mean"], normalization["summary_std"])[0]
            batch_points = torch.from_numpy(points[None, :, :]).to(device)
            batch_mask = torch.ones((1, points.shape[0]), dtype=torch.bool, device=device)
            batch_summary = torch.from_numpy(summary[None, :].astype(np.float32)).to(device)
            output = model(batch_points, batch_mask, batch_summary)
            decoded_norm = output["decoded"][0].detach().cpu().numpy().astype(np.float32)
            decoded = denormalize(decoded_norm, normalization["point_mean"], normalization["point_std"])
            z = output["z"][0].detach().cpu().numpy().astype(np.float32)
            image_name = f"particle_{idx + 1:02d}_p{int(row['particle_id']):06d}_original_decoded.png"
            image_path = args.assets / image_name
            plot_one(row, raw_points, decoded, z, image_path)
            err = chamfer_xyte(raw_points, decoded)
            item = {"row": row, "original": raw_points, "decoded": decoded, "z": z, "image": image_path}
            items.append(item)
            summary = {
                "rank": idx + 1,
                "source_path": str(row["source_path"]),
                "particle_id": int(row["particle_id"]),
                "n_hits": int(row["n_hits"]),
                "view_n_hits": int(row["view_n_hits"]),
                "sample_fraction": float(row["sample_fraction"]),
                "energy_sum": float(row["energy_sum"]),
                "time_span": float(row["time_span"]),
                "decoded_points": int(decoded.shape[0]),
                "latent_dim": int(z.shape[0]),
                "reconstruction_chamfer_xyte_feature": err,
                "image": image_path.as_posix(),
            }
            for latent_idx, value in enumerate(z.tolist()):
                summary[f"z_{latent_idx:02d}"] = float(value)
            summary_rows.append(summary)

    montage = args.assets / "top_largest_original_decoded_montage.png"
    plot_montage(items, montage)
    csv_path = args.assets / "largest_reconstruction_summary.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    (args.assets / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    try:
        asset_rel_dir = args.assets.relative_to(args.report.parent)
    except ValueError:
        asset_rel_dir = args.assets

    lines = [
        "# Phase 2 Largest Particle Reconstruction Snapshots",
        "",
        "This demo uses the corrected `voxel_corner_min2_v001` particle dataset and the selected Phase 2 reconstruction checkpoint.",
        "",
        f"- Dataset: `{args.dataset.as_posix()}`",
        f"- Checkpoint: `{args.checkpoint.as_posix()}`",
        f"- Model: `{model_config.backbone}` / `{model_config.objective}`",
        f"- Latent size: `{model_config.latent_dim}`",
        f"- Decoder output size: `{model_config.decoder_points}` points",
        f"- Numeric summary: [{csv_path.name}]({(asset_rel_dir / csv_path.name).as_posix()})",
        "",
        f"The left panel in each image is the actual Phase 2 input view, capped at 512 sampled hits for very large particles. The middle panel is the decoded {model_config.decoder_points}-point point set. The right panel is the encoded latent vector `z`. Coordinates are particle-relative Phase 2 features: centered `x`, centered `y`, and scaled relative time. Color is normalized `log1p(ToT)` energy.",
        "",
        "Important reading: the autoencoder is a compact point-set summarizer here. It is not expected to reproduce every hit of a 1000+ hit component; the useful question is whether the latent preserves enough morphology for downstream family grouping.",
        "",
        "## Montage",
        "",
        f"![Top largest original decoded montage]({(asset_rel_dir / montage.name).as_posix()})",
        "",
        "## Top Largest Views",
        "",
        "| Rank | Source | Particle | Hits | View hits | Sample | Chamfer | z preview | Image |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for summary in summary_rows:
        z_preview = ", ".join(f"{summary[f'z_{i:02d}']:.3f}" for i in range(min(4, model_config.latent_dim)))
        rel_image = Path(summary["image"]).relative_to(args.report.parent)
        lines.append(
            f"| {summary['rank']} | `{summary['source_path']}` | {summary['particle_id']} | "
            f"{summary['n_hits']} | {summary['view_n_hits']} | {summary['sample_fraction']:.3f} | "
            f"{summary['reconstruction_chamfer_xyte_feature']:.4f} | `{z_preview}` | "
            f"[png]({rel_image.as_posix()}) |"
        )
    lines.extend([""])
    for summary in summary_rows:
        rel_image = Path(summary["image"]).relative_to(args.report.parent)
        lines.extend(
            [
                f"### Rank {summary['rank']}: Particle {summary['particle_id']}",
                "",
                f"![Particle {summary['particle_id']}]({rel_image.as_posix()})",
                "",
            ]
        )
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": args.report.as_posix(), "assets": args.assets.as_posix(), "rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()

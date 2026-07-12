#!/usr/bin/env python3
"""Generate diagnostics for a canonical path autoencoder checkpoint."""

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

from particle_classification.experiments.autoencoders.path import (
    energy_path_tensor,
    load_path_ae_checkpoint,
    path_ae_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def eval_split(model, arr: np.ndarray, device: str) -> dict[str, float]:
    metrics = []
    with torch.no_grad():
        for start in range(0, arr.shape[0], 512):
            target = torch.from_numpy(arr[start : start + 512]).to(device=device, dtype=torch.float32)
            recon = model(target)["reconstruction"]
            _, row = path_ae_loss(recon, target)
            metrics.append(row)
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


def plot_path_pair(path: np.ndarray, recon: np.ndarray, output: Path, title: str) -> None:
    fig = plt.figure(figsize=(13.5, 4.2), constrained_layout=True)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3)
    for ax, arr, subtitle in [(ax1, path, "canonical target"), (ax2, recon, "decoded")]:
        color = np.clip(arr[:, 3], 0.0, 1.0)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=color, cmap="viridis", s=16, linewidths=0.0)
        ax.set_title(subtitle)
        ax.set_xlabel("x canonical")
        ax.set_ylabel("y canonical")
        ax.set_zlabel("t norm")
        ax.view_init(elev=24, azim=-58)
    u = np.linspace(0.0, 1.0, path.shape[0])
    ax3.plot(u, path[:, 3], label="target energy", linewidth=2)
    ax3.plot(u, recon[:, 3], label="decoded energy", linewidth=1.5)
    ax3.set_xlabel("path rank")
    ax3.set_ylabel("energy")
    ax3.legend()
    fig.suptitle(title)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = load_path_ae_checkpoint(args.checkpoint, device=device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint["model_config"]

    split_arrays = {}
    split_metrics = {}
    for split in ["val", "test"]:
        with np.load(args.cache / f"{split}.npz", allow_pickle=False) as data:
            arr = data["path"].astype(np.float32)
        split_arrays[split] = arr
        split_metrics[split] = eval_split(model, arr[: min(arr.shape[0], 16384)], device=device)

    metadata = pd.read_csv(args.cache / "test_metadata.csv")
    test = split_arrays["test"]
    with torch.no_grad():
        target = torch.from_numpy(test).to(device=device, dtype=torch.float32)
        chunks = []
        energy_errors = []
        for start in range(0, test.shape[0], 512):
            item = target[start : start + 512]
            recon = model(item)["reconstruction"]
            err = torch.sqrt(torch.mean((energy_path_tensor(recon) - energy_path_tensor(item)).pow(2), dim=(1, 2)) + 1e-6) / torch.sqrt(
                torch.mean(energy_path_tensor(item).pow(2), dim=(1, 2)) + 1e-6
            )
            energy_errors.append(err.detach().cpu().numpy())
            chunks.append(recon.detach().cpu().numpy())
        recon_all = np.concatenate(chunks, axis=0)
        errors = np.concatenate(energy_errors, axis=0)

    metadata = metadata.iloc[: test.shape[0]].copy()
    metadata["energy_path_relative_l2"] = errors
    largest = metadata.sort_values("n_hits", ascending=False).head(args.top).copy()
    rows = []
    for rank, (idx, row) in enumerate(largest.iterrows(), start=1):
        image = args.assets / f"largest_{rank:02d}_p{int(row['particle_id'])}_path_ae.png"
        plot_path_pair(test[idx], recon_all[idx], image, f"rank {rank}: {row['source_path']} p{row['particle_id']} n={row['n_hits']}")
        rows.append({**row.to_dict(), "rank": rank, "image": image.as_posix()})
    pd.DataFrame(rows).to_csv(args.assets / "largest_path_ae_summary.csv", index=False)

    # Error histogram.
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(errors, bins=80, color="#2563eb", alpha=0.85)
    ax.axvline(0.1, color="#dc2626", linestyle="--", label="target 0.1")
    ax.set_xlabel("energy path relative L2")
    ax.set_ylabel("test particles")
    ax.legend()
    fig.savefig(args.assets / "energy_path_error_hist.png", dpi=160)
    plt.close(fig)

    try:
        rel_assets = args.assets.relative_to(args.report.parent)
    except ValueError:
        rel_assets = args.assets
    lines = [
        "# Phase 2 Canonical Path Autoencoder Report",
        "",
        "This model separates detector-plane rotation before encoding. The network sees canonical ordered paths and stores `theta_xy` as metadata outside the latent vector.",
        "",
        f"- checkpoint: `{args.checkpoint.as_posix()}`",
        f"- latent dim: `{model_config['latent_dim']}`",
        f"- path samples: `{model_config['path_points']}`",
        f"- hidden dim: `{model_config['hidden_dim']}`",
        "",
        "## Metrics",
        "",
        "| split | loss | path relative L2 | energy-path relative L2 | diff relative L2 | energy sum rel L1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in split_metrics.items():
        lines.append(
            f"| {split} | {metrics['loss']:.4f} | {metrics['path_relative_l2']:.4f} | "
            f"{metrics['energy_path_relative_l2']:.4f} | {metrics['diff_relative_l2']:.4f} | "
            f"{metrics['energy_sum_relative_l1']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"![energy path error histogram]({(rel_assets / 'energy_path_error_hist.png').as_posix()})",
            "",
            "## Largest Test Particles",
            "",
            "| rank | source | particle | hits | energy-path rel L2 | image |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        image = Path(row["image"])
        try:
            image_rel = image.relative_to(args.report.parent)
        except ValueError:
            image_rel = image
        lines.append(
            f"| {row['rank']} | `{row['source_path']}` | {row['particle_id']} | {row['n_hits']} | "
            f"{row['energy_path_relative_l2']:.4f} | [png]({image_rel.as_posix()}) |"
        )
    lines.append("")
    for row in rows:
        image = Path(row["image"])
        try:
            image_rel = image.relative_to(args.report.parent)
        except ValueError:
            image_rel = image
        lines.extend([f"### Rank {row['rank']}", "", f"![rank {row['rank']}]({image_rel.as_posix()})", ""])
    args.report.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "checkpoint": args.checkpoint.as_posix(),
        "report": args.report.as_posix(),
        "split_metrics": split_metrics,
        "test_energy_path_relative_l2_mean": float(np.mean(errors)),
        "test_energy_path_relative_l2_median": float(np.median(errors)),
        "test_energy_path_relative_l2_p95": float(np.percentile(errors, 95)),
        "target_met_mean_lt_0_1": bool(np.mean(errors) < 0.1),
    }
    (args.assets / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

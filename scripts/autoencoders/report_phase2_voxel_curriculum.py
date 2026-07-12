#!/usr/bin/env python3
"""Generate a markdown report for a Phase 2 voxel-AE curriculum run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="voxel_autoencoder_summary.json")
    parser.add_argument("--out", type=Path, required=True, help="Markdown report path.")
    parser.add_argument("--asset-dir", type=Path, default=Path("experimental_notes/assets/phase2_voxel_autoencoder/curriculum_training"))
    parser.add_argument("--title", default="Phase 2 Voxel Autoencoder Curriculum Training")
    return parser.parse_args()


def fmt(value: object, precision: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def make_plots(metrics: pd.DataFrame, stage_summary: pd.DataFrame, asset_dir: Path) -> dict[str, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for column in ["loss", "val_loss", "mse", "val_mse", "occupied_mse", "val_occupied_mse"]:
            if column in metrics:
                ax.plot(metrics["step"], metrics[column], label=column)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("MSE")
        ax.set_title("Reconstruction losses")
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = asset_dir / "loss_curves.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["loss_curves"] = path

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(metrics["step"], metrics["lr"], color="tab:blue", label="LR")
        ax.set_xlabel("training step")
        ax.set_ylabel("learning rate")
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_title("Learning rate schedule")
        path = asset_dir / "lr_schedule.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["lr_schedule"] = path

        fig, ax2 = plt.subplots(figsize=(9, 4.8))
        for column, color in [("noise_std", "tab:orange"), ("voxel_dropout", "tab:green"), ("blur_mix", "tab:red")]:
            if column in metrics:
                ax2.plot(metrics["step"], metrics[column], color=color, alpha=0.8, label=column)
        ax2.set_xlabel("training step")
        ax2.set_ylabel("corruption amount")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.25)
        ax2.set_title("Input corruption schedule")
        path = asset_dir / "corruption_schedule.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["corruption_schedule"] = path

    if not stage_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(stage_summary["stage"], stage_summary["actual_steps"], color="tab:purple", alpha=0.75)
        ax.set_xlabel("phase")
        ax.set_ylabel("actual steps")
        ax.set_title("Phase length after plateau checks")
        ax.tick_params(axis="x", rotation=70)
        ax.grid(axis="y", alpha=0.25)
        path = asset_dir / "phase_steps.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["phase_steps"] = path

    return plots


def rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    run = summary["run"]
    metrics_path = Path(run["metrics"])
    stage_path = Path(run["stage_summary"])
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    stage_summary = pd.read_csv(stage_path) if stage_path.exists() else pd.DataFrame()
    plots = make_plots(metrics, stage_summary, args.asset_dir)

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# {args.title}")
    lines.append("")
    lines.append("This report documents the staged 3D voxel autoencoder training run over centered `[T,Y,X] = [32,64,64]` energy tensors.")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- summary: `{args.summary.as_posix()}`")
    lines.append(f"- checkpoint: `{run['checkpoint']}`")
    if run.get("init_checkpoint") or summary.get("init_checkpoint"):
        lines.append(f"- init checkpoint: `{run.get('init_checkpoint') or summary.get('init_checkpoint')}`")
    lines.append(f"- metrics CSV: `{run['metrics']}`")
    lines.append(f"- stage summary CSV: `{run['stage_summary']}`")
    lines.append(f"- duration: `{fmt(run.get('duration_s'), 2)} s`")
    best_label = "loss" if run.get("best_val_loss") is not None else "MSE"
    best_value = run.get("best_val_loss", run.get("best_val_mse"))
    lines.append(f"- best validation {best_label}: `{fmt(best_value)}` at step `{run.get('best_step')}`")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    model = summary["model_config"]
    train = summary["train_config"]
    grid = summary["grid_config"]
    lines.append("| item | value |")
    lines.append("|---|---:|")
    for key in [
        "architecture",
        "shape_latent_dim",
        "aux_latent_dim",
        "hidden_dim",
        "patch_hidden_dim",
        "patch_embed_dim",
        "output_activation",
        "output_bias_init",
    ]:
        lines.append(f"| model `{key}` | `{model[key]}` |")
    for key in ["t_bins", "y_bins", "x_bins", "time_bin", "xy_bin"]:
        lines.append(f"| grid `{key}` | `{grid[key]}` |")
    for key in [
        "batch_size",
        "learning_rate",
        "min_learning_rate",
        "weight_decay",
        "occupied_weight",
        "energy_weight",
        "lr_schedule",
        "phase_patience_evals",
        "lr_patience_evals",
        "keep_lr_across_stages",
        "stop_lr_below",
    ]:
        lines.append(f"| train `{key}` | `{train[key]}` |")
    lines.append("")

    if not stage_summary.empty:
        if "best_val_loss" not in stage_summary and "best_val_mse" in stage_summary:
            stage_summary = stage_summary.rename(columns={"best_val_mse": "best_val_loss"})
        display = stage_summary[
            [
                "stage",
                "actual_steps",
                "stop_reason",
                "best_val_loss",
                "blur_kernel",
                "blur_mix",
                "noise_std",
                "voxel_dropout",
                "duration_s",
            ]
        ].copy()
        for column in ["best_val_loss", "blur_mix", "noise_std", "voxel_dropout", "duration_s"]:
            display[column] = display[column].map(lambda value: fmt(float(value)))
        lines.append("## Phase Summary")
        lines.append("")
        lines.append(dataframe_to_markdown(display))
        lines.append("")

    lines.append("## Test Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key, value in run.items():
        if key.startswith("test_"):
            lines.append(f"| `{key}` | `{fmt(value)}` |")
    lines.append("")

    if plots:
        lines.append("## Training Plots")
        lines.append("")
        for label, path in plots.items():
            lines.append(f"![{label}]({rel(path, out_dir)})")
            lines.append("")

    if not metrics.empty:
        tail = metrics.tail(8).copy()
        for column in tail.columns:
            if pd.api.types.is_float_dtype(tail[column]):
                tail[column] = tail[column].map(lambda value: fmt(float(value)))
        lines.append("## Last Evaluations")
        lines.append("")
        lines.append(dataframe_to_markdown(tail))
        lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    sys.exit(main())

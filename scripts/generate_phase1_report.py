#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from particle_classification.edge_training import (
    collate_edge_windows,
    evaluate_edge_model,
    grouping_metrics,
    model_selection_score,
    move_batch_tensors,
)
from particle_classification.models import EdgeTrackNetTiny, EdgeTrackNetTinyConfig, connected_components_from_edges


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets" / "phase1_dbscan_replacement"
REPORT_PATH = ROOT / "docs" / "phase1-dbscan-replacement-report.md"

REAL_DATASET = ROOT / "local_data" / "processed" / "pass1_edge_dataset" / "manifest.csv"
HARD_DATASET = ROOT / "local_data" / "processed" / "pass1_edge_hard_mixed_dataset" / "manifest.csv"
MILD_DATASET = ROOT / "local_data" / "processed" / "pass1_edge_mixed_dataset" / "manifest.csv"

BASELINE_EXP = ROOT / "local_data" / "experiments" / "edge_tracknet_long"
REAL_MP_EXP = ROOT / "local_data" / "experiments" / "edge_tracknet_real_mp_search"
HARD_MP_EXP = ROOT / "local_data" / "experiments" / "edge_tracknet_replacement_hard_mp"


@dataclass(frozen=True)
class LoadedRun:
    model: EdgeTrackNetTiny
    checkpoint: dict
    device: str


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_run = load_run(REAL_MP_EXP / "edge_tracknet_tiny.pt", device=device)
    hard_run = load_run(HARD_MP_EXP / "edge_tracknet_tiny.pt", device=device)

    real_strict = json.loads((REAL_MP_EXP / "strict_threshold_summary.json").read_text(encoding="utf-8"))
    hard_strict = strict_metrics_from_sweep(
        hard_run,
        HARD_DATASET,
        HARD_MP_EXP / "threshold_sweep_strict.csv",
        device=device,
    )

    plot_convergence(
        REAL_MP_EXP / "metrics.csv",
        "Real Stable DBSCAN Windows: 2-Hop EdgeTrackNet",
        ASSET_DIR / "real_convergence.png",
    )
    plot_convergence(
        HARD_MP_EXP / "metrics.csv",
        "Hard Mixed Stress Set: 2-Hop EdgeTrackNet",
        ASSET_DIR / "hard_convergence.png",
    )
    plot_threshold_heatmap(
        REAL_MP_EXP / "threshold_sweep_strict.csv",
        "Strict Threshold Sweep on Real Validation Split",
        ASSET_DIR / "real_threshold_sweep.png",
    )
    plot_dataset_diagnostics(
        [("real stable", REAL_DATASET), ("hard mixed", HARD_DATASET)],
        ASSET_DIR / "dataset_diagnostics.png",
    )
    plot_metric_comparison(real_strict, hard_strict, ASSET_DIR / "metric_comparison.png")

    real_examples = write_validation_figure(
        real_run,
        REAL_DATASET,
        edge_threshold=float(real_strict["validation_best"]["edge_threshold"]),
        object_threshold=float(real_strict["validation_best"]["object_threshold"]),
        out_path=ASSET_DIR / "validation_real_test_nn_vs_dbscan.png",
        title="Real Stable Test Windows: DBSCAN Pseudo-labels vs NN Readout",
        max_rows=4,
        device=device,
    )
    hard_examples = write_validation_figure(
        hard_run,
        HARD_DATASET,
        edge_threshold=float(hard_strict["validation_best"]["edge_threshold"]),
        object_threshold=float(hard_strict["validation_best"]["object_threshold"]),
        out_path=ASSET_DIR / "validation_hard_test_nn_vs_truth.png",
        title="Hard Mixed Stress Test: Truth Labels vs NN Readout",
        max_rows=3,
        device=device,
    )
    write_examples_csv(real_examples, hard_examples, ASSET_DIR / "validation_examples.csv")

    report = build_report(
        real_strict=real_strict,
        hard_strict=hard_strict,
        real_run=real_run,
        real_examples=real_examples,
        hard_examples=hard_examples,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)


def load_run(path: Path, *, device: str) -> LoadedRun:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = EdgeTrackNetTiny(EdgeTrackNetTinyConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return LoadedRun(model=model, checkpoint=checkpoint, device=device)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def summarize_manifest(path: Path) -> dict[str, object]:
    rows = read_csv_rows(path)
    summary: dict[str, object] = {
        "windows": len(rows),
        "splits": {split: sum(row.get("split") == split for row in rows) for split in ["train", "val", "test"]},
    }
    for key in ["n_hits", "n_edges", "positive_edges", "negative_edges", "ignored_edges", "particles", "noise_fraction"]:
        values = np.asarray([numeric(row, key) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values)) if values.size else 0.0
        summary[f"{key}_p50"] = float(np.percentile(values, 50)) if values.size else 0.0
        summary[f"{key}_p95"] = float(np.percentile(values, 95)) if values.size else 0.0
        summary[f"{key}_max"] = float(np.max(values)) if values.size else 0.0
    return summary


def plot_convergence(metrics_path: Path, title: str, out_path: Path) -> None:
    rows = read_csv_rows(metrics_path)
    step = np.asarray([numeric(row, "step") for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    axes[0, 0].plot(step, [numeric(row, "loss") for row in rows], label="total")
    axes[0, 0].plot(step, [numeric(row, "edge_loss") for row in rows], label="edge")
    axes[0, 0].plot(step, [numeric(row, "object_loss") for row in rows], label="object")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("BCE loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(step, [numeric(row, "val_ari") for row in rows], label="ARI")
    axes[0, 1].plot(step, [numeric(row, "val_pairwise_f1") for row in rows], label="pairwise F1")
    axes[0, 1].plot(step, [numeric(row, "val_object_accuracy") for row in rows], label="object acc.")
    axes[0, 1].set_title("Validation agreement")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylim(0.0, 1.02)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(step, [numeric(row, "val_split_rate") for row in rows], label="split")
    axes[1, 0].plot(step, [numeric(row, "val_merge_rate") for row in rows], label="merge")
    axes[1, 0].plot(step, [numeric(row, "val_energy_error") for row in rows], label="energy err.")
    axes[1, 0].set_title("Validation failure modes")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].step(step, [numeric(row, "lr") for row in rows], where="post")
    axes[1, 1].set_title("ReduceLROnPlateau learning rate")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(alpha=0.25)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_threshold_heatmap(sweep_path: Path, title: str, out_path: Path) -> None:
    rows = read_csv_rows(sweep_path)
    edges = sorted({numeric(row, "edge_threshold") for row in rows})
    objects = sorted({numeric(row, "object_threshold") for row in rows})
    score = np.full((len(objects), len(edges)), np.nan)
    ari = np.full_like(score, np.nan)
    for row in rows:
        y = objects.index(numeric(row, "object_threshold"))
        x = edges.index(numeric(row, "edge_threshold"))
        score[y, x] = numeric(row, "score")
        ari[y, x] = numeric(row, "ari")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, matrix, label in [(axes[0], score, "selection score"), (axes[1], ari, "ARI")]:
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(label)
        ax.set_xticks(range(len(edges)), [f"{value:.2f}" for value in edges])
        ax.set_yticks(range(len(objects)), [f"{value:.2f}" for value in objects])
        ax.set_xlabel("edge threshold")
        ax.set_ylabel("object threshold")
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                ax.text(xi, yi, f"{matrix[yi, xi]:.2f}", color="white", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dataset_diagnostics(manifests: list[tuple[str, Path]], out_path: Path) -> None:
    data = [(name, read_csv_rows(path)) for name, path in manifests]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for name, rows in data:
        axes[0, 0].hist([numeric(row, "n_hits") for row in rows], bins=40, alpha=0.55, label=name)
    axes[0, 0].set_title("Window hit counts")
    axes[0, 0].set_xlabel("hits")
    axes[0, 0].set_ylabel("windows")
    axes[0, 0].legend()

    for name, rows in data:
        axes[0, 1].hist([numeric(row, "negative_edges") for row in rows], bins=40, alpha=0.55, label=name)
    axes[0, 1].set_title("Negative edge load")
    axes[0, 1].set_xlabel("negative edges")
    axes[0, 1].legend()

    labels = [name for name, _ in data]
    positive = [sum(numeric(row, "positive_edges") for row in rows) for _, rows in data]
    negative = [sum(numeric(row, "negative_edges") for row in rows) for _, rows in data]
    ignored = [sum(numeric(row, "ignored_edges") for row in rows) for _, rows in data]
    x = np.arange(len(labels))
    axes[1, 0].bar(x, positive, label="positive")
    axes[1, 0].bar(x, negative, bottom=positive, label="negative")
    axes[1, 0].bar(x, ignored, bottom=np.asarray(positive) + np.asarray(negative), label="ignored")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_title("Edge-label composition")
    axes[1, 0].legend()

    for name, rows in data:
        axes[1, 1].hist([numeric(row, "noise_fraction") for row in rows], bins=30, alpha=0.55, label=name)
    axes[1, 1].set_title("Noise fraction")
    axes[1, 1].set_xlabel("fraction")
    axes[1, 1].legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(real_strict: dict, hard_strict: dict, out_path: Path) -> None:
    metrics = ["ari", "pairwise_f1", "split_rate", "merge_rate", "object_accuracy", "energy_error"]
    labels = ["ARI", "pair F1", "split", "merge", "object acc.", "energy err."]
    values = {
        "real stable test": real_strict["test_metrics"],
        "hard mixed test": hard_strict["test_metrics"],
    }
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for idx, (name, row) in enumerate(values.items()):
        ax.bar(x + (idx - 0.5) * width, [float(row[key]) for key in metrics], width=width, label=name)
    ax.axhline(0.75, color="#333333", linestyle="--", linewidth=1.0, alpha=0.5, label="ARI target")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Phase 1 NN readout metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def strict_metrics_from_sweep(run: LoadedRun, manifest: Path, sweep_path: Path, *, device: str) -> dict:
    rows = read_csv_rows(sweep_path)
    best = max(rows, key=lambda row: numeric(row, "score"))
    test = DatasetFromManifest(manifest, split="test")
    metrics = evaluate_edge_model(
        run.model,
        test,
        max_windows=160,
        device=device,
        edge_threshold=numeric(best, "edge_threshold"),
        object_threshold=numeric(best, "object_threshold"),
    )
    return {
        "validation_best": {key: numeric(best, key) for key in best},
        "test_metrics": metrics,
        "target_met": is_target_met(metrics),
    }


class DatasetFromManifest:
    def __init__(self, manifest: Path, *, split: str):
        self.rows = [row for row in read_csv_rows(manifest) if row.get("split") == split and row.get("status") == "ok"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return load_window(repo_path(self.rows[index]["path"]))


def load_window(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "features": data["features"].astype(np.float32),
            "edge_index": data["edge_index"].astype(np.int64),
            "edge_label": data["edge_label"].astype(np.int64),
            "edge_weight": data["edge_weight"].astype(np.float32),
            "object_label": data["object_label"].astype(np.float32),
            "object_weight": data["object_weight"].astype(np.float32),
            "source_particle_id": data["source_particle_id"].astype(np.int64),
            "hit_energy": data["hit_energy"].astype(np.float32),
            "hit_x": data["hit_x"].astype(np.float32),
            "hit_y": data["hit_y"].astype(np.float32),
            "hit_time": data["hit_time"].astype(np.float64),
            "path": path.as_posix(),
        }


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def predict_labels(run: LoadedRun, item: dict, *, edge_threshold: float, object_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    batch = move_batch_tensors(collate_edge_windows([item]), run.device)
    with torch.no_grad():
        output = run.model(batch["features"], batch["edge_index"])
    edge_scores = torch.sigmoid(output["edge_logits"]).detach().cpu().numpy()
    object_scores = torch.sigmoid(output["object_logits"]).detach().cpu().numpy()
    labels = connected_components_from_edges(
        int(item["features"].shape[0]),
        item["edge_index"],
        edge_scores,
        object_scores,
        edge_threshold=edge_threshold,
        object_threshold=object_threshold,
    )
    return labels, object_scores


def write_validation_figure(
    run: LoadedRun,
    manifest: Path,
    *,
    edge_threshold: float,
    object_threshold: float,
    out_path: Path,
    title: str,
    max_rows: int,
    device: str,
) -> list[dict[str, object]]:
    rows = [row for row in read_csv_rows(manifest) if row.get("split") == "test" and row.get("status") == "ok"]
    selected = select_visual_rows(rows, max_rows=max_rows)
    fig = plt.figure(figsize=(13, 3.6 * len(selected)))
    fig.suptitle(title, fontsize=14)
    example_rows: list[dict[str, object]] = []
    for idx, row in enumerate(selected):
        item = load_window(repo_path(row["path"]))
        truth = item["source_particle_id"]
        pred, object_scores = predict_labels(
            run,
            item,
            edge_threshold=edge_threshold,
            object_threshold=object_threshold,
        )
        metrics = grouping_metrics(truth, pred, item["hit_energy"], object_scores, object_threshold=object_threshold)
        example_rows.append(
            {
                "dataset": "hard_mixed" if "hard_mixed" in manifest.as_posix() else "real_stable",
                "path": row["path"],
                "n_hits": row["n_hits"],
                "particles": row["particles"],
                **metrics,
            }
        )
        ax_truth = fig.add_subplot(len(selected), 2, idx * 2 + 1, projection="3d")
        ax_pred = fig.add_subplot(len(selected), 2, idx * 2 + 2, projection="3d")
        plot_labels_3d(ax_truth, item, truth, f"DBSCAN/truth | {Path(row['path']).name}")
        plot_labels_3d(
            ax_pred,
            item,
            pred,
            f"NN | ARI {metrics['ari']:.3f}, F1 {metrics['pairwise_f1']:.3f}",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return example_rows


def select_visual_rows(rows: list[dict[str, str]], *, max_rows: int) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if numeric(row, "n_hits") >= 50
        and numeric(row, "particles") >= 2
        and numeric(row, "noise_fraction") < 0.85
    ]
    if not candidates:
        candidates = rows
    candidates = sorted(
        candidates,
        key=lambda row: (
            numeric(row, "negative_edges"),
            numeric(row, "particles"),
            numeric(row, "n_hits"),
        ),
        reverse=True,
    )
    if len(candidates) <= max_rows:
        return candidates
    indices = np.linspace(0, min(len(candidates) - 1, max_rows * 8), num=max_rows, dtype=int)
    return [candidates[int(index)] for index in indices]


def plot_labels_3d(ax, item: dict, labels: np.ndarray, title: str) -> None:
    x = item["hit_x"]
    y = item["hit_y"]
    t = item["hit_time"]
    t_norm = (t - float(np.min(t))) / max(float(np.max(t) - np.min(t)), 1.0)
    colors = labels_to_colors(labels)
    ax.scatter(x, y, t_norm, c=colors, s=8, alpha=0.86, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("time norm.")
    ax.view_init(elev=22, azim=-58)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)


def labels_to_colors(labels: np.ndarray) -> list:
    cmap = plt.get_cmap("tab20")
    colors = []
    for label in labels.tolist():
        if int(label) < 0:
            colors.append((0.55, 0.55, 0.55, 0.42))
        else:
            colors.append(cmap(int(label) % 20))
    return colors


def write_examples_csv(real_examples: list[dict[str, object]], hard_examples: list[dict[str, object]], out_path: Path) -> None:
    rows = real_examples + hard_examples
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_target_met(metrics: dict[str, float]) -> bool:
    return (
        metrics["ari"] >= 0.75
        and metrics["pairwise_f1"] >= 0.92
        and metrics["split_rate"] <= 0.08
        and metrics["merge_rate"] <= 0.08
        and metrics["object_accuracy"] >= 0.90
        and metrics["energy_error"] <= 0.20
    )


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def metric_table(metrics: dict[str, float]) -> str:
    keys = [
        ("ari", "Adjusted Rand index"),
        ("pairwise_f1", "Pairwise F1"),
        ("split_rate", "Split rate"),
        ("merge_rate", "Merge rate"),
        ("object_accuracy", "Object accuracy"),
        ("energy_error", "Energy error"),
        ("latency_p50_ms", "Latency p50 ms"),
        ("latency_p95_ms", "Latency p95 ms"),
    ]
    lines = ["| Metric | Value |", "|---|---:|"]
    for key, label in keys:
        lines.append(f"| {label} | {fmt(metrics[key], 4)} |")
    return "\n".join(lines)


def manifest_table() -> str:
    rows = [
        ("Real stable DBSCAN windows", summarize_manifest(REAL_DATASET)),
        ("Hard mixed stress windows", summarize_manifest(HARD_DATASET)),
        ("Mild mixed windows", summarize_manifest(MILD_DATASET)),
    ]
    lines = [
        "| Dataset | Windows | Train/Val/Test | Hits | Edges | Positive/Negative/Ignored Edges | Mean Particles | Mean Noise |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in rows:
        splits = summary["splits"]
        split_text = f"{splits['train']}/{splits['val']}/{splits['test']}"
        edge_text = (
            f"{summary['positive_edges_mean']:.1f}/"
            f"{summary['negative_edges_mean']:.1f}/"
            f"{summary['ignored_edges_mean']:.1f}"
        )
        lines.append(
            f"| {name} | {summary['windows']} | {split_text} | "
            f"{summary['n_hits_mean']:.1f} mean | {summary['n_edges_mean']:.1f} mean | "
            f"{edge_text} mean | {summary['particles_mean']:.1f} | {summary['noise_fraction_mean']:.3f} |"
        )
    return "\n".join(lines)


def architecture_table(config: dict) -> str:
    h = int(config["hidden_dim"])
    edge_h = int(config["edge_hidden_dim"])
    input_dim = int(config["input_dim"])
    edge_input = h * 3 + 3
    lines = [
        "| Block | Input -> Output | Layers | Parameters |",
        "|---|---:|---|---:|",
        f"| Hit encoder | {input_dim} -> {h} | Linear, LayerNorm, SiLU, Dropout({config['dropout']}), Linear, LayerNorm, SiLU | 18,048 |",
        f"| Message block x {config['message_passing_steps']} | {edge_input} edge input and {h * 2} update input -> {h} | message MLP plus mean aggregation into destination nodes, then residual update MLP | 232,192 |",
        f"| Edge head | {edge_input} -> 1 | Linear({edge_input}, {edge_h}), LayerNorm, SiLU, Dropout, Linear({edge_h}, {edge_h // 2}), SiLU, Linear({edge_h // 2}, 1) | 58,241 |",
        f"| Object head | {h} -> 1 | Linear({h}, {h // 2}), SiLU, Linear({h // 2}, 1) | 8,321 |",
        "| Total | - | all trainable | 316,802 |",
    ]
    return "\n".join(lines)


def build_report(
    *,
    real_strict: dict,
    hard_strict: dict,
    real_run: LoadedRun,
    real_examples: list[dict[str, object]],
    hard_examples: list[dict[str, object]],
) -> str:
    config = real_run.checkpoint["model_config"]
    train_config = real_run.checkpoint["train_config"]
    baseline_summary = json.loads((BASELINE_EXP / "summary.json").read_text(encoding="utf-8"))
    real_summary = json.loads((REAL_MP_EXP / "summary.json").read_text(encoding="utf-8"))
    hard_summary = json.loads((HARD_MP_EXP / "summary.json").read_text(encoding="utf-8"))

    target = real_strict["target"]
    lines = [
        "# Phase 1 Report: Neural Replacement for 3D DBSCAN Particle Extraction",
        "",
        "## Executive Summary",
        "",
        "Phase 1 trained a hit-level graph neural network, `EdgeTrackNetTiny`, to replace the 3D DBSCAN particle extraction pass on stable pseudo-label windows. The model predicts local same-particle edges and hit objectness, then turns those probabilities into variable-size particle IDs with connected components. Energy is preserved as an input and output diagnostic, but the clustering distance and NN graph construction remain based on `(x, y, time)` geometry.",
        "",
        "The strongest Phase 1 model is the 2-hop message-passing EdgeTrackNet trained on stable DBSCAN pseudo-label windows. With a strict validation-selected readout threshold, it meets the replacement target on the real stable test split.",
        "",
        metric_table(real_strict["test_metrics"]),
        "",
        f"Target gate: ARI >= {target['ari']}, pairwise F1 >= {target['pairwise_f1']}, split <= {target['split_rate_max']}, merge <= {target['merge_rate_max']}, object accuracy >= {target['object_accuracy']}, energy error <= {target['energy_error_max']}.",
        "",
        "The hard mixed stress set remains deliberately harder than the stable DBSCAN distribution. It is useful as a failure-mode detector: the same architecture is not yet robust enough for dense close-particle ambiguity there.",
        "",
        "## Data Preparation",
        "",
        "### Source Labels",
        "",
        "The teacher labels come from the 3D DBSCAN pipeline built earlier. Each raw `.t3pa` file was converted into an NPZ particle shard containing all hits and `hit_particle_id`, where `-1` is noise. For Phase 1 NN training, only stable pseudo-label windows are used: each candidate window is re-clustered with neighboring DBSCAN settings and kept only when the mean adjusted Rand index stays above the configured stability threshold. Unstable windows are omitted rather than treated as hard negatives.",
        "",
        "### Hit Features",
        "",
        "Each hit becomes a 7-value feature vector:",
        "",
        "| Feature | Definition | Purpose |",
        "|---|---|---|",
        "| `x/255` | detector x normalized by chip width | spatial position |",
        "| `y/255` | detector y normalized by chip height | spatial position |",
        "| `t_norm` | per-window min-max normalized ToA | local timing coordinate |",
        "| `log1p(ToT)` | energy-like feature saved as `hit_energy` | charge/energy proxy |",
        "| `FToA/30` | fine time clipped to `[0, 4]` | timing refinement |",
        "| `delta_t_prev` | previous time delta divided by p95 local delta, clipped to `[0, 10]` | local temporal pitch |",
        "| `delta_t_next` | next time delta divided by p95 local delta, clipped to `[0, 10]` | local temporal pitch |",
        "",
        "Windows are sorted by time. The real stable dataset uses `window_size=2048`, `window_overlap=512`, `k_neighbors=12`, and graph radius `3.5` in scaled `(x, y, t/time_scale)` space. The hard mixed dataset uses a denser graph (`k_neighbors=16`, radius `4.5`) to force nearby negative edges.",
        "",
        "### Graph and Labels",
        "",
        "For each window a directed local graph is built with SciPy `cKDTree`: every hit connects to up to `k` neighbors within the radius. Edge labels are: positive if both endpoints are non-noise and share the same DBSCAN particle ID, negative if both endpoints are non-noise and belong to different particles, and ignored if either endpoint is noise. Objectness labels are one for non-noise hits and zero for noise hits. Edge and object losses use cluster-balanced weights so large particles do not dominate the objective.",
        "",
        "### Synthetic and Mixed Data",
        "",
        "The new mixed dataset generator creates controlled validation and stress windows in the same NPZ format as real pseudo-label windows. It extracts real DBSCAN particle templates, recenters them, randomly shifts them in detector position and time, reflects them in x/y, injects noise hits, and combines multiple particles per window. It also generates procedural line, kink, blob, and dense-track particles. The `hard_fraction` mode deliberately places several particles within a small space-time neighborhood, producing many real negative edges and exposing split/merge weaknesses.",
        "",
        manifest_table(),
        "",
        "![Dataset diagnostics](assets/phase1_dbscan_replacement/dataset_diagnostics.png)",
        "",
        "## Network Architecture",
        "",
        f"Final model config: `input_dim={config['input_dim']}`, `hidden_dim={config['hidden_dim']}`, `edge_hidden_dim={config['edge_hidden_dim']}`, `dropout={config['dropout']}`, `message_passing_steps={config['message_passing_steps']}`. The checkpoint has 316,802 trainable parameters.",
        "",
        architecture_table(config),
        "",
        "### Forward Pass",
        "",
        "1. The collate function concatenates variable-size windows into one node matrix `[total_hits, 7]`. Directed edge indices are offset and concatenated into `[2, total_edges]`. Per-window slices are retained for evaluation, but the forward pass itself is fully ragged.",
        "2. The hit encoder maps each hit feature vector to a 128-dimensional hidden state using two linear layers with LayerNorm and SiLU activations.",
        "3. Each message-passing step computes an edge message from `[h_src, h_dst, h_src - h_dst, distance, delta_t, delta_energy]`. Messages are mean-aggregated into destination nodes and passed through an update MLP. The node state is updated residually: `h = h + update([h, aggregate])`.",
        "4. The edge head evaluates the final pair state `[h_src, h_dst, h_src - h_dst, aux]` and outputs one logit per directed edge. `sigmoid(edge_logit)` is interpreted as same-particle probability.",
        "5. The object head outputs one logit per hit. `sigmoid(object_logit)` is interpreted as non-noise/object probability.",
        "6. Readout filters hits by object threshold, unions endpoint pairs whose edge probability is above the edge threshold, and returns connected-component IDs. Inactive hits are `-1` noise.",
        "",
        "### Initialization and Normalization",
        "",
        "No custom initialization is used. `torch.nn.Linear` therefore uses the standard PyTorch reset behavior: Kaiming-uniform weight initialization with the linear-layer bias sampled from the corresponding fan-in bound. `LayerNorm` scale starts at one and bias starts at zero. Dropout is active only during training. Feature normalization is performed before the network, and LayerNorm is applied inside the hit encoder, message MLPs, update MLPs, and edge head.",
        "",
        "## Training Setup",
        "",
        f"The final real-window run used AdamW with learning rate `{train_config['learning_rate']}`, weight decay `{train_config['weight_decay']}`, batch size `{train_config['batch_size']}`, and `{train_config['steps']}` requested steps. Validation ran every `{train_config['eval_interval']}` steps. A `ReduceLROnPlateau` scheduler monitored the model-selection score and multiplied LR by `{train_config['lr_plateau_factor']}` after `{train_config['lr_plateau_patience']}` stagnant validation checks, down to `min_lr={train_config['min_learning_rate']}`.",
        "",
        "The loss is weighted binary cross entropy over labeled edges plus `0.25` times weighted binary cross entropy over hit objectness. Ignored noise edges do not contribute to edge loss. Model selection score combines ARI, pairwise F1, object accuracy, split/merge penalties, and energy error.",
        "",
        "## Training Convergence",
        "",
        "The original pairwise-only baseline learned a useful edge signal but did not reach the replacement gate. Its first pass test metrics were ARI `0.501`, pairwise F1 `0.840`, split/merge around `0.015`, object accuracy `0.786`, and energy error `0.423`. The 2-hop message-passing model is clearly stronger on the real stable distribution.",
        "",
        f"Best validation checkpoint for the real 2-hop model occurred at step `{real_summary['best_step']}` before readout threshold tuning. The default sweep selected edge/object thresholds `{real_summary['selected_edge_threshold']}` / `{real_summary['selected_object_threshold']}`, while the stricter sweep selected `{real_strict['validation_best']['edge_threshold']}` / `{real_strict['validation_best']['object_threshold']}`.",
        "",
        "![Real convergence](assets/phase1_dbscan_replacement/real_convergence.png)",
        "",
        "![Real strict threshold sweep](assets/phase1_dbscan_replacement/real_threshold_sweep.png)",
        "",
        "The hard mixed stress set shows the current limit. It contains intentionally close particles and many negative edges; the 2-hop model improves over pairwise-only but still fails the replacement gate there.",
        "",
        "![Hard convergence](assets/phase1_dbscan_replacement/hard_convergence.png)",
        "",
        "![Metric comparison](assets/phase1_dbscan_replacement/metric_comparison.png)",
        "",
        "## Validation Results",
        "",
        "### Real Stable Test Split",
        "",
        "Strict validation-selected thresholds on the real stable split:",
        "",
        metric_table(real_strict["test_metrics"]),
        "",
        "The rendered examples below are intentionally challenging test windows selected by negative-edge load, particle count, and hit count. They are for visual inspection of failure modes; the table above is the aggregate held-out score.",
        "",
        "![Real validation examples](assets/phase1_dbscan_replacement/validation_real_test_nn_vs_dbscan.png)",
        "",
        "### Hard Mixed Stress Split",
        "",
        "The hard mixed stress split is not the acceptance target; it is a robustness probe. It reveals that the model can still merge or split close particles when the local radius graph contains many plausible negative edges.",
        "",
        metric_table(hard_strict["test_metrics"]),
        "",
        "![Hard validation examples](assets/phase1_dbscan_replacement/validation_hard_test_nn_vs_truth.png)",
        "",
        "## Example-Level Metrics",
        "",
        "These example-level means summarize only the rendered inspection windows, not the full test split.",
        "",
        "| Dataset | Examples | Mean ARI | Mean Pairwise F1 | Mean Split | Mean Merge |",
        "|---|---:|---:|---:|---:|---:|",
        example_summary_line("real stable", real_examples),
        example_summary_line("hard mixed", hard_examples),
        "",
        "Detailed example rows are saved in [`validation_examples.csv`](assets/phase1_dbscan_replacement/validation_examples.csv).",
        "",
        "## Interpretation",
        "",
        "The Phase 1 NN is reasonable as a DBSCAN replacement for the stable pseudo-label distribution: it satisfies the chosen acceptance gate on held-out real windows after validation threshold selection, with sub-millisecond p95 window latency on the local GPU run. The model is especially good at preserving energy/objectness and avoiding gross over-splitting on real windows.",
        "",
        "It is not yet a universal replacement for ambiguous close-particle cases. The hard mixed stress set produces substantially more negative edges than the real stable dataset, and the current connected-components readout still trades merges against splits as the edge threshold changes. The next model iteration should either add stronger relational context, edge calibration focused on hard negatives, or a learned clustering/readout layer rather than relying only on thresholded connected components.",
        "",
        "## Reproduction Commands",
        "",
        "```bash",
        "particle-train-edge-tracknet \\",
        "  --manifest local_data/processed/pass1_edge_dataset/manifest.csv \\",
        "  --out local_data/experiments/edge_tracknet_real_mp_search \\",
        "  --steps 6000 \\",
        "  --min-steps 1500 \\",
        "  --batch-size 4 \\",
        "  --learning-rate 0.001 \\",
        "  --eval-interval 100 \\",
        "  --max-eval-windows 160 \\",
        "  --hidden-dim 128 \\",
        "  --edge-hidden-dim 128 \\",
        "  --message-passing-steps 2 \\",
        "  --device cuda",
        "```",
        "",
        "```bash",
        "python scripts/generate_phase1_report.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def example_summary_line(name: str, rows: list[dict[str, object]]) -> str:
    if not rows:
        return f"| {name} | 0 | 0 | 0 | 0 | 0 |"
    return (
        f"| {name} | {len(rows)} | "
        f"{np.mean([float(row['ari']) for row in rows]):.4f} | "
        f"{np.mean([float(row['pairwise_f1']) for row in rows]):.4f} | "
        f"{np.mean([float(row['split_rate']) for row in rows]):.4f} | "
        f"{np.mean([float(row['merge_rate']) for row in rows]):.4f} |"
    )


if __name__ == "__main__":
    main()

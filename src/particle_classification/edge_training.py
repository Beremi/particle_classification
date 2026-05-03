from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .data.edge_training import (
    EdgeDatasetConfig,
    build_edge_training_set,
    discover_particle_shards,
    load_manifest_rows,
    load_particle_shard,
)
from .data.particles import adjusted_rand_index, write_dict_rows
from .models import EdgeTrackNetTiny, EdgeTrackNetTinyConfig, connected_components_from_edges


@dataclass(frozen=True)
class EdgeTrainConfig:
    seed: int = 20260503
    steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    edge_hidden_dim: int = 64
    dropout: float = 0.05
    object_loss_weight: float = 0.25
    eval_interval: int = 100
    max_eval_windows: int = 96
    edge_threshold: float = 0.5
    object_threshold: float = 0.5


class EdgeWindowDataset(Dataset):
    def __init__(self, manifest_path: str | Path, *, split: str):
        self.rows = load_manifest_rows(manifest_path, split=split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        row = self.rows[index]
        with np.load(row["path"], allow_pickle=False) as data:
            return {
                "features": data["features"].astype(np.float32),
                "edge_index": data["edge_index"].astype(np.int64),
                "edge_label": data["edge_label"].astype(np.int64),
                "edge_weight": data["edge_weight"].astype(np.float32),
                "object_label": data["object_label"].astype(np.float32),
                "object_weight": data["object_weight"].astype(np.float32),
                "source_particle_id": data["source_particle_id"].astype(np.int64),
                "hit_energy": data["hit_energy"].astype(np.float32),
                "path": row["path"],
            }


def collate_edge_windows(items: list[dict[str, np.ndarray | str]]) -> dict[str, object]:
    features = []
    edge_indices = []
    edge_labels = []
    edge_weights = []
    object_labels = []
    object_weights = []
    truth_labels = []
    energies = []
    node_slices = [0]
    edge_slices = [0]
    node_offset = 0
    for item in items:
        item_features = np.asarray(item["features"], dtype=np.float32)
        item_edge_index = np.asarray(item["edge_index"], dtype=np.int64).copy()
        item_edge_index += node_offset
        features.append(item_features)
        edge_indices.append(item_edge_index)
        edge_labels.append(np.asarray(item["edge_label"], dtype=np.int64))
        edge_weights.append(np.asarray(item["edge_weight"], dtype=np.float32))
        object_labels.append(np.asarray(item["object_label"], dtype=np.float32))
        object_weights.append(np.asarray(item["object_weight"], dtype=np.float32))
        truth_labels.append(np.asarray(item["source_particle_id"], dtype=np.int64))
        energies.append(np.asarray(item["hit_energy"], dtype=np.float32))
        node_offset += item_features.shape[0]
        node_slices.append(node_offset)
        edge_slices.append(edge_slices[-1] + item_edge_index.shape[1])

    if edge_indices:
        edge_index = np.concatenate(edge_indices, axis=1)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return {
        "features": torch.from_numpy(np.concatenate(features, axis=0)),
        "edge_index": torch.from_numpy(edge_index),
        "edge_label": torch.from_numpy(np.concatenate(edge_labels, axis=0)),
        "edge_weight": torch.from_numpy(np.concatenate(edge_weights, axis=0)),
        "object_label": torch.from_numpy(np.concatenate(object_labels, axis=0)),
        "object_weight": torch.from_numpy(np.concatenate(object_weights, axis=0)),
        "truth_labels": truth_labels,
        "energies": energies,
        "node_slices": node_slices,
        "edge_slices": edge_slices,
        "paths": [str(item["path"]) for item in items],
    }


def edge_tracknet_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, object],
    *,
    object_loss_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    edge_label = batch["edge_label"].to(output["edge_logits"].device)
    edge_weight = batch["edge_weight"].to(output["edge_logits"].device)
    edge_mask = edge_label >= 0
    if bool(torch.any(edge_mask)):
        edge_targets = edge_label[edge_mask].to(dtype=output["edge_logits"].dtype)
        edge_logits = output["edge_logits"][edge_mask]
        edge_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            edge_logits,
            edge_targets,
            reduction="none",
        )
        weights = edge_weight[edge_mask].to(dtype=edge_loss_raw.dtype)
        edge_loss = torch.sum(edge_loss_raw * weights) / torch.clamp(torch.sum(weights), min=1e-6)
    else:
        edge_loss = output["edge_logits"].sum() * 0.0

    object_label = batch["object_label"].to(output["object_logits"].device)
    object_weight = batch["object_weight"].to(output["object_logits"].device)
    object_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
        output["object_logits"],
        object_label,
        reduction="none",
    )
    object_loss = torch.sum(object_loss_raw * object_weight) / torch.clamp(torch.sum(object_weight), min=1e-6)
    total = edge_loss + object_loss_weight * object_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "edge_loss": float(edge_loss.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
    }


def train_edge_tracknet(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: EdgeTrainConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, object]:
    config = config or EdgeTrainConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_dataset = EdgeWindowDataset(manifest_path, split="train")
    val_dataset = EdgeWindowDataset(manifest_path, split="val")
    test_dataset = EdgeWindowDataset(manifest_path, split="test")
    if not train_dataset:
        raise RuntimeError("No train windows found in edge training dataset.")

    model_config = EdgeTrackNetTinyConfig(
        input_dim=7,
        hidden_dim=config.hidden_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        dropout=config.dropout,
    )
    model = EdgeTrackNetTiny(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    metrics_rows: list[dict[str, object]] = []
    for step in range(1, config.steps + 1):
        items = random_batch(train_dataset, config.batch_size)
        batch = move_batch_tensors(collate_edge_windows(items), device)
        model.train()
        result = model(batch["features"], batch["edge_index"])
        loss, loss_metrics = edge_tracknet_loss(result, batch, object_loss_weight=config.object_loss_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % config.eval_interval == 0 or step == config.steps:
            val_metrics = evaluate_edge_model(
                model,
                val_dataset if len(val_dataset) else train_dataset,
                max_windows=config.max_eval_windows,
                device=device,
                edge_threshold=config.edge_threshold,
                object_threshold=config.object_threshold,
            )
            row = {"step": step, **loss_metrics, **{f"val_{key}": value for key, value in val_metrics.items()}}
            metrics_rows.append(row)
            write_dict_rows(output / "metrics.csv", metrics_rows)
            if verbose:
                print(
                    f"step {step}/{config.steps}: loss={row['loss']:.4f}, "
                    f"val_ari={row['val_ari']:.3f}, val_pairwise_f1={row['val_pairwise_f1']:.3f}",
                    flush=True,
                )

    test_metrics = evaluate_edge_model(
        model,
        test_dataset if len(test_dataset) else val_dataset if len(val_dataset) else train_dataset,
        max_windows=config.max_eval_windows,
        device=device,
        edge_threshold=config.edge_threshold,
        object_threshold=config.object_threshold,
    )
    checkpoint_path = output / "edge_tracknet_tiny.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    summary = {
        "checkpoint": checkpoint_path.as_posix(),
        "steps": config.steps,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "test_metrics": test_metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_experiment_report(output / "report.md", summary, metrics_rows)
    return summary


def random_batch(dataset: EdgeWindowDataset, batch_size: int) -> list[dict[str, np.ndarray | str]]:
    if len(dataset) <= batch_size:
        return [dataset[idx] for idx in range(len(dataset))]
    indices = random.sample(range(len(dataset)), k=batch_size)
    return [dataset[idx] for idx in indices]


def move_batch_tensors(batch: dict[str, object], device: str) -> dict[str, object]:
    out = dict(batch)
    for key in ["features", "edge_index", "edge_label", "edge_weight", "object_label", "object_weight"]:
        out[key] = out[key].to(device)
    return out


@torch.no_grad()
def evaluate_edge_model(
    model: EdgeTrackNetTiny,
    dataset: EdgeWindowDataset,
    *,
    max_windows: int,
    device: str,
    edge_threshold: float,
    object_threshold: float,
) -> dict[str, float]:
    model.eval()
    n_eval = min(len(dataset), max_windows)
    if n_eval == 0:
        return {
            "ari": 0.0,
            "pairwise_f1": 0.0,
            "split_rate": 0.0,
            "merge_rate": 0.0,
            "object_accuracy": 0.0,
            "energy_error": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
        }
    indices = np.linspace(0, len(dataset) - 1, num=n_eval, dtype=int)
    rows = []
    latencies = []
    for idx in indices.tolist():
        item = dataset[idx]
        batch = move_batch_tensors(collate_edge_windows([item]), device)
        start = time.perf_counter()
        output = model(batch["features"], batch["edge_index"])
        latencies.append((time.perf_counter() - start) * 1000.0)
        edge_scores = torch.sigmoid(output["edge_logits"]).detach().cpu().numpy()
        object_scores = torch.sigmoid(output["object_logits"]).detach().cpu().numpy()
        truth = np.asarray(item["source_particle_id"], dtype=np.int64)
        pred = connected_components_from_edges(
            truth.shape[0],
            np.asarray(item["edge_index"]),
            edge_scores,
            object_scores,
            edge_threshold=edge_threshold,
            object_threshold=object_threshold,
        )
        rows.append(grouping_metrics(truth, pred, np.asarray(item["hit_energy"], dtype=np.float32), object_scores))

    return {
        "ari": float(np.mean([row["ari"] for row in rows])),
        "pairwise_f1": float(np.mean([row["pairwise_f1"] for row in rows])),
        "split_rate": float(np.mean([row["split_rate"] for row in rows])),
        "merge_rate": float(np.mean([row["merge_rate"] for row in rows])),
        "object_accuracy": float(np.mean([row["object_accuracy"] for row in rows])),
        "energy_error": float(np.mean([row["energy_error"] for row in rows])),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def grouping_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    energy: np.ndarray,
    object_scores: np.ndarray,
) -> dict[str, float]:
    return {
        "ari": adjusted_rand_index(truth, pred),
        "pairwise_f1": pairwise_f1(truth, pred),
        "split_rate": split_rate(truth, pred),
        "merge_rate": merge_rate(truth, pred),
        "object_accuracy": float(np.mean((object_scores >= 0.5) == (truth >= 0))) if truth.size else 0.0,
        "energy_error": energy_conservation_error(truth, pred, energy),
    }


def pairwise_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    truth_pairs = cluster_pair_count(truth)
    pred_pairs = cluster_pair_count(pred)
    if truth_pairs == 0 and pred_pairs == 0:
        return 1.0
    tp = 0.0
    for truth_label in set(int(label) for label in truth.tolist()) - {-1}:
        mask = truth == truth_label
        tp += cluster_pair_count(pred[mask])
    precision = tp / pred_pairs if pred_pairs else 0.0
    recall = tp / truth_pairs if truth_pairs else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def cluster_pair_count(labels: np.ndarray) -> float:
    total = 0.0
    for label in set(int(item) for item in labels.tolist()) - {-1}:
        count = int(np.sum(labels == label))
        total += count * (count - 1) / 2.0
    return total


def split_rate(truth: np.ndarray, pred: np.ndarray) -> float:
    truth_ids = sorted(set(int(label) for label in truth.tolist()) - {-1})
    if not truth_ids:
        return 0.0
    splits = 0
    for truth_id in truth_ids:
        pred_ids = set(int(label) for label in pred[truth == truth_id].tolist()) - {-1}
        if len(pred_ids) > 1:
            splits += 1
    return splits / len(truth_ids)


def merge_rate(truth: np.ndarray, pred: np.ndarray) -> float:
    pred_ids = sorted(set(int(label) for label in pred.tolist()) - {-1})
    if not pred_ids:
        return 0.0
    merges = 0
    for pred_id in pred_ids:
        truth_ids = set(int(label) for label in truth[pred == pred_id].tolist()) - {-1}
        if len(truth_ids) > 1:
            merges += 1
    return merges / len(pred_ids)


def energy_conservation_error(truth: np.ndarray, pred: np.ndarray, energy: np.ndarray) -> float:
    truth_energy = float(np.sum(energy[truth >= 0]))
    pred_energy = float(np.sum(energy[pred >= 0]))
    if truth_energy <= 1e-6:
        return 0.0 if pred_energy <= 1e-6 else 1.0
    return abs(pred_energy - truth_energy) / max(truth_energy, 1e-6)


def write_preprocessing_audit(
    particle_dir: str | Path,
    output_dir: str | Path,
    *,
    max_shards: int = 256,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shards = discover_particle_shards(particle_dir)[:max_shards]
    rows = []
    for shard in shards:
        data = load_particle_shard(shard.path)
        x = np.asarray(data["hit_x"])
        y = np.asarray(data["hit_y"])
        t = np.asarray(data["hit_time"])
        tot = np.asarray(data["hit_tot"])
        pairs = set(zip(x.astype(int).tolist(), y.astype(int).tolist(), strict=True))
        rows.append(
            {
                "source_path": shard.source_path,
                "hits": int(x.shape[0]),
                "duplicate_pixels": int(x.shape[0] - len(pairs)),
                "tot_mean": float(np.mean(tot)) if tot.size else 0.0,
                "tot_p95": float(np.percentile(tot, 95)) if tot.size else 0.0,
                "time_span": float(np.max(t) - np.min(t)) if t.size else 0.0,
            }
        )
    write_dict_rows(output / "preprocessing_audit.csv", rows)
    summary = {
        "shards": len(rows),
        "hits": int(sum(row["hits"] for row in rows)),
        "median_hits": float(np.median([row["hits"] for row in rows])) if rows else 0.0,
        "median_time_span": float(np.median([row["time_span"] for row in rows])) if rows else 0.0,
        "mean_duplicate_pixels": float(np.mean([row["duplicate_pixels"] for row in rows])) if rows else 0.0,
    }
    (output / "preprocessing_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_stability_report(dataset_manifest: str | Path, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(dataset_manifest)
    ari_values = [float(row["stability_ari"]) for row in rows if row.get("stability_ari")]
    summary = {
        "windows": len(rows),
        "mean_stability_ari": float(np.mean(ari_values)) if ari_values else 0.0,
        "p10_stability_ari": float(np.percentile(ari_values, 10)) if ari_values else 0.0,
        "p50_stability_ari": float(np.percentile(ari_values, 50)) if ari_values else 0.0,
        "p90_stability_ari": float(np.percentile(ari_values, 90)) if ari_values else 0.0,
    }
    lines = [
        "# DBSCAN Pseudo-label Stability",
        "",
        f"- windows: {summary['windows']}",
        f"- mean ARI: {summary['mean_stability_ari']:.4f}",
        f"- p10/p50/p90 ARI: {summary['p10_stability_ari']:.4f} / {summary['p50_stability_ari']:.4f} / {summary['p90_stability_ari']:.4f}",
        "",
        "Stable windows are used as pseudo-labels; unstable windows are omitted from this first pass.",
    ]
    (output / "dbscan_stability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "dbscan_stability_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_edge_preliminary_experiment(
    particle_dir: str | Path,
    params_path: str | Path,
    dataset_dir: str | Path,
    experiment_dir: str | Path,
    *,
    dataset_config: EdgeDatasetConfig | None = None,
    train_config: EdgeTrainConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, object]:
    experiment = Path(experiment_dir)
    experiment.mkdir(parents=True, exist_ok=True)
    audit = write_preprocessing_audit(particle_dir, experiment)
    dataset_result = build_edge_training_set(
        particle_dir,
        dataset_dir,
        params_path=params_path,
        config=dataset_config,
        verbose=verbose,
    )
    stability = write_stability_report(Path(dataset_dir) / "manifest.csv", experiment)
    training = train_edge_tracknet(
        Path(dataset_dir) / "manifest.csv",
        experiment,
        config=train_config,
        device=device,
        verbose=verbose,
    )
    summary = {"audit": audit, "dataset": dataset_result, "stability": stability, "training": training}
    (experiment / "preliminary_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def write_experiment_report(path: Path, summary: dict[str, object], metrics_rows: list[dict[str, object]]) -> None:
    test_metrics = summary.get("test_metrics", {})
    lines = [
        "# EdgeTrackNet-Tiny Preliminary Experiment",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- steps: {summary['steps']}",
        f"- train/val/test windows: {summary['train_windows']} / {summary['val_windows']} / {summary['test_windows']}",
        "",
        "## Test Metrics",
        "",
    ]
    if isinstance(test_metrics, dict):
        for key, value in test_metrics.items():
            lines.append(f"- `{key}`: {float(value):.6f}")
    if metrics_rows:
        last = metrics_rows[-1]
        lines.extend(["", "## Last Validation Snapshot", ""])
        for key, value in last.items():
            if key == "step":
                lines.append(f"- `step`: {value}")
            elif isinstance(value, (int, float)):
                lines.append(f"- `{key}`: {float(value):.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

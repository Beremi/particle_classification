from __future__ import annotations

import csv
import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import (
    EDGE_ATTR_NAMES,
    EdgeDatasetConfig,
    build_edge_training_set,
    discover_particle_shards,
    load_manifest_rows,
    load_particle_shard,
)
from ...dbscan.pipeline import adjusted_rand_index, write_dict_rows
from .model import EdgeTrackNetTiny, EdgeTrackNetTinyConfig, connected_components_from_edges


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
    message_passing_steps: int = 0
    object_loss_weight: float = 0.25
    edge_loss: str = "focal"
    focal_gamma: float = 2.0
    embedding_loss_weight: float = 0.0
    eval_interval: int = 100
    max_eval_windows: int = 96
    edge_threshold: float = 0.5
    object_threshold: float = 0.5
    lr_plateau_patience: int = 4
    lr_plateau_factor: float = 0.5
    min_learning_rate: float = 1e-5
    min_steps: int = 0
    target_ari: float = 0.75
    target_pairwise_f1: float = 0.92
    target_split_rate: float = 0.08
    target_merge_rate: float = 0.08
    target_object_accuracy: float = 0.90
    target_energy_error: float = 0.20
    threshold_sweep: bool = True


class EdgeWindowDataset(Dataset):
    def __init__(self, manifest_path: str | Path, *, split: str):
        self.rows = load_manifest_rows(manifest_path, split=split)
        self.indices_by_curriculum_source: dict[str, list[int]] = {}
        for idx, row in enumerate(self.rows):
            source = row.get("curriculum_source", "")
            if source:
                self.indices_by_curriculum_source.setdefault(source, []).append(idx)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str]:
        row = self.rows[index]
        with np.load(row["path"], allow_pickle=False) as data:
            item = {
                "features": data["features"].astype(np.float32),
                "edge_index": data["edge_index"].astype(np.int64),
                "edge_label": data["edge_label"].astype(np.int64),
                "edge_weight": data["edge_weight"].astype(np.float32),
                "object_label": data["object_label"].astype(np.float32),
                "object_weight": data["object_weight"].astype(np.float32),
                "source_particle_id": data["source_particle_id"].astype(np.int64),
                "hit_energy": data["hit_energy"].astype(np.float32),
                "path": row["path"],
                "curriculum_source": row.get("curriculum_source", ""),
                "label_source": row.get("label_source", ""),
            }
            if "edge_attr" in data.files:
                item["edge_attr"] = data["edge_attr"].astype(np.float32)
            if "edge_type" in data.files:
                item["edge_type"] = data["edge_type"].astype(np.uint8)
            return item


def collate_edge_windows(items: list[dict[str, np.ndarray | str]]) -> dict[str, object]:
    features = []
    edge_indices = []
    edge_labels = []
    edge_weights = []
    edge_attrs = []
    object_labels = []
    object_weights = []
    truth_labels = []
    energies = []
    truth_tensors = []
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
        if "edge_attr" in item:
            edge_attrs.append(np.asarray(item["edge_attr"], dtype=np.float32))
        object_labels.append(np.asarray(item["object_label"], dtype=np.float32))
        object_weights.append(np.asarray(item["object_weight"], dtype=np.float32))
        item_truth = np.asarray(item["source_particle_id"], dtype=np.int64)
        truth_labels.append(item_truth)
        truth_tensors.append(item_truth)
        energies.append(np.asarray(item["hit_energy"], dtype=np.float32))
        node_offset += item_features.shape[0]
        node_slices.append(node_offset)
        edge_slices.append(edge_slices[-1] + item_edge_index.shape[1])

    if edge_indices:
        edge_index = np.concatenate(edge_indices, axis=1)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    out = {
        "features": torch.from_numpy(np.concatenate(features, axis=0)),
        "edge_index": torch.from_numpy(edge_index),
        "edge_label": torch.from_numpy(np.concatenate(edge_labels, axis=0)),
        "edge_weight": torch.from_numpy(np.concatenate(edge_weights, axis=0)),
        "object_label": torch.from_numpy(np.concatenate(object_labels, axis=0)),
        "object_weight": torch.from_numpy(np.concatenate(object_weights, axis=0)),
        "truth_tensor": torch.from_numpy(np.concatenate(truth_tensors, axis=0)),
        "truth_labels": truth_labels,
        "energies": energies,
        "node_slices": node_slices,
        "edge_slices": edge_slices,
        "paths": [str(item["path"]) for item in items],
    }
    if edge_attrs and len(edge_attrs) == len(items):
        out["edge_attr"] = torch.from_numpy(np.concatenate(edge_attrs, axis=0))
    return out


def edge_tracknet_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, object],
    *,
    object_loss_weight: float = 0.25,
    edge_loss: str = "focal",
    focal_gamma: float = 2.0,
    embedding_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    edge_label = batch["edge_label"].to(output["edge_logits"].device)
    edge_weight = batch["edge_weight"].to(output["edge_logits"].device)
    edge_mask = edge_label >= 0
    if bool(torch.any(edge_mask)):
        edge_targets = edge_label[edge_mask].to(dtype=output["edge_logits"].dtype)
        edge_logits = output["edge_logits"][edge_mask]
        if edge_loss == "focal":
            bce = torch.nn.functional.binary_cross_entropy_with_logits(edge_logits, edge_targets, reduction="none")
            probs = torch.sigmoid(edge_logits)
            p_t = probs * edge_targets + (1.0 - probs) * (1.0 - edge_targets)
            edge_loss_raw = bce * torch.pow(1.0 - p_t, focal_gamma)
        elif edge_loss == "bce":
            edge_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                edge_logits,
                edge_targets,
                reduction="none",
            )
        else:
            raise ValueError(f"Unknown edge loss: {edge_loss}")
        weights = edge_weight[edge_mask].to(dtype=edge_loss_raw.dtype)
        if "edge_attr" in batch:
            edge_attr = batch["edge_attr"].to(output["edge_logits"].device)[edge_mask]
            close_negative = (edge_targets < 0.5) & (edge_attr[:, 5] <= 3.5)
            weights = torch.where(close_negative, weights * 2.0, weights)
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
    embedding_loss = metric_embedding_loss(output, batch) if embedding_loss_weight > 0.0 else edge_loss * 0.0
    total = edge_loss + object_loss_weight * object_loss + embedding_loss_weight * embedding_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "edge_loss": float(edge_loss.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
        "embedding_loss": float(embedding_loss.detach().cpu()),
    }


def train_edge_tracknet(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: EdgeTrainConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
    init_checkpoint: str | Path | None = None,
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

    schema = edge_dataset_schema(manifest_path, train_dataset)
    model_config = EdgeTrackNetTinyConfig(
        input_dim=int(schema["input_dim"]),
        edge_attr_dim=int(schema["edge_attr_dim"]),
        hidden_dim=config.hidden_dim,
        edge_hidden_dim=config.edge_hidden_dim,
        dropout=config.dropout,
        message_passing_steps=config.message_passing_steps,
    )
    model = EdgeTrackNetTiny(model_config).to(device)
    init_checkpoint_path = Path(init_checkpoint) if init_checkpoint is not None else None
    if init_checkpoint_path is not None:
        load_initial_checkpoint(model, model_config, init_checkpoint_path, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.min_learning_rate,
    )
    eval_dataset = val_dataset if len(val_dataset) else train_dataset
    best_score = float("-inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] = {}
    stop_reason = "max_steps"

    metrics_rows: list[dict[str, object]] = []
    last_step = 0
    for step in range(1, config.steps + 1):
        last_step = step
        items = random_batch(train_dataset, config.batch_size)
        batch = move_batch_tensors(collate_edge_windows(items), device)
        model.train()
        result = model(batch["features"], batch["edge_index"], batch.get("edge_attr"))
        loss, loss_metrics = edge_tracknet_loss(
            result,
            batch,
            object_loss_weight=config.object_loss_weight,
            edge_loss=config.edge_loss,
            focal_gamma=config.focal_gamma,
            embedding_loss_weight=config.embedding_loss_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % config.eval_interval == 0 or step == config.steps:
            val_metrics = evaluate_edge_model(
                model,
                eval_dataset,
                max_windows=config.max_eval_windows,
                device=device,
                edge_threshold=config.edge_threshold,
                object_threshold=config.object_threshold,
            )
            score = model_selection_score(val_metrics)
            scheduler.step(score)
            lr = float(optimizer.param_groups[0]["lr"])
            if score > best_score:
                best_score = score
                best_step = step
                best_val_metrics = dict(val_metrics)
                best_state = copy.deepcopy(model.state_dict())
            row = {
                "step": step,
                "lr": lr,
                **loss_metrics,
                "val_score": score,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            metrics_rows.append(row)
            write_dict_rows(output / "metrics.csv", metrics_rows)
            if verbose:
                print(
                    f"step {step}/{config.steps}: loss={row['loss']:.4f}, "
                    f"val_ari={row['val_ari']:.3f}, val_pairwise_f1={row['val_pairwise_f1']:.3f}, "
                    f"lr={lr:.2e}",
                    flush=True,
                )
            if step >= max(config.min_steps, 1) and is_reasonable_edge_model(val_metrics, config):
                stop_reason = "target_reached"
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    selected_edge_threshold = config.edge_threshold
    selected_object_threshold = config.object_threshold
    threshold_summary: dict[str, object] = {"enabled": False}
    if config.threshold_sweep and len(eval_dataset):
        sweep = sweep_edge_thresholds(
            model,
            eval_dataset,
            output / "threshold_sweep.csv",
            max_windows=config.max_eval_windows,
            device=device,
        )
        selected_edge_threshold = float(sweep["best"]["edge_threshold"])
        selected_object_threshold = float(sweep["best"]["object_threshold"])
        threshold_summary = sweep

    test_metrics = evaluate_edge_model(
        model,
        test_dataset if len(test_dataset) else val_dataset if len(val_dataset) else train_dataset,
        max_windows=config.max_eval_windows,
        device=device,
        edge_threshold=selected_edge_threshold,
        object_threshold=selected_object_threshold,
    )
    checkpoint_path = output / "edge_tracknet_tiny.pt"
    normalization_path = Path(manifest_path).parent / "normalization.json"
    normalization_data = (
        json.loads(normalization_path.read_text(encoding="utf-8")) if normalization_path.exists() else None
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(config),
            "init_checkpoint": init_checkpoint_path.as_posix() if init_checkpoint_path is not None else "",
            "normalization": normalization_data,
            "normalization_path": normalization_path.as_posix() if normalization_path.exists() else "",
            "best_step": best_step,
            "best_val_score": best_score,
            "best_val_metrics": best_val_metrics,
            "selected_edge_threshold": selected_edge_threshold,
            "selected_object_threshold": selected_object_threshold,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    summary = {
        "checkpoint": checkpoint_path.as_posix(),
        "init_checkpoint": init_checkpoint_path.as_posix() if init_checkpoint_path is not None else "",
        "steps": last_step,
        "requested_steps": config.steps,
        "stop_reason": stop_reason,
        "best_step": best_step,
        "best_val_score": best_score,
        "best_val_metrics": best_val_metrics,
        "selected_edge_threshold": selected_edge_threshold,
        "selected_object_threshold": selected_object_threshold,
        "target_met": is_reasonable_edge_model(test_metrics, config),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "test_windows": len(test_dataset),
        "threshold_sweep": threshold_summary,
        "test_metrics": test_metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_experiment_report(output / "report.md", summary, metrics_rows)
    return summary


def random_batch(dataset: EdgeWindowDataset, batch_size: int) -> list[dict[str, np.ndarray | str]]:
    groups = {key: value for key, value in dataset.indices_by_curriculum_source.items() if value}
    if len(groups) >= 2 and batch_size > 1:
        sources = sorted(groups)
        selected_indices: list[int] = []
        base = batch_size // len(sources)
        remainder = batch_size % len(sources)
        for source_idx, source in enumerate(sources):
            draws = base + (1 if source_idx < remainder else 0)
            selected_indices.extend(random.choice(groups[source]) for _ in range(draws))
        random.shuffle(selected_indices)
        return [dataset[idx] for idx in selected_indices]
    if len(dataset) <= batch_size:
        return [dataset[idx] for idx in range(len(dataset))]
    indices = random.sample(range(len(dataset)), k=batch_size)
    return [dataset[idx] for idx in indices]


def load_initial_checkpoint(
    model: EdgeTrackNetTiny,
    model_config: EdgeTrackNetTinyConfig,
    checkpoint_path: Path,
    *,
    device: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = EdgeTrackNetTinyConfig(**checkpoint["model_config"])
    expected = asdict(model_config)
    observed = asdict(checkpoint_config)
    if observed != expected:
        raise ValueError(
            "Initial checkpoint model config does not match requested training config: "
            f"checkpoint={observed} requested={expected}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])


def move_batch_tensors(batch: dict[str, object], device: str) -> dict[str, object]:
    out = dict(batch)
    for key in [
        "features",
        "edge_index",
        "edge_attr",
        "edge_label",
        "edge_weight",
        "object_label",
        "object_weight",
        "truth_tensor",
    ]:
        if key in out:
            out[key] = out[key].to(device)
    return out


def edge_dataset_schema(manifest_path: str | Path, dataset: EdgeWindowDataset) -> dict[str, int]:
    norm_path = Path(manifest_path).parent / "normalization.json"
    if norm_path.exists():
        data = json.loads(norm_path.read_text(encoding="utf-8"))
        return {
            "input_dim": int(data.get("input_dim", 7)),
            "edge_attr_dim": int(data.get("edge_attr_dim", 3)),
        }
    if len(dataset):
        item = dataset[0]
        return {
            "input_dim": int(np.asarray(item["features"]).shape[1]),
            "edge_attr_dim": int(np.asarray(item["edge_attr"]).shape[1]) if "edge_attr" in item else 3,
        }
    return {"input_dim": 7, "edge_attr_dim": 3}


def metric_embedding_loss(output: dict[str, torch.Tensor], batch: dict[str, object]) -> torch.Tensor:
    embeddings = output["node_embedding"]
    edge_index = batch["edge_index"].to(embeddings.device)
    edge_label = batch["edge_label"].to(embeddings.device)
    if edge_index.shape[1] == 0:
        return embeddings.sum() * 0.0
    src = edge_index[0].long()
    dst = edge_index[1].long()
    distances = torch.linalg.norm(embeddings[src] - embeddings[dst], dim=-1)
    positive = edge_label == 1
    negative = edge_label == 0
    pieces = []
    if bool(torch.any(positive)):
        pieces.append(torch.mean(distances[positive] ** 2))
    if bool(torch.any(negative)):
        margin = 1.0
        neg_loss = torch.relu(margin - distances[negative]) ** 2
        if "edge_attr" in batch:
            edge_attr = batch["edge_attr"].to(embeddings.device)
            close = edge_attr[negative, 5] <= 3.5
            neg_loss = torch.where(close, neg_loss * 2.0, neg_loss)
        pieces.append(torch.mean(neg_loss))
    if not pieces:
        return embeddings.sum() * 0.0
    return torch.stack(pieces).mean()


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
        output = model(batch["features"], batch["edge_index"], batch.get("edge_attr"))
        latencies.append((time.perf_counter() - start) * 1000.0)
        edge_scores = torch.sigmoid(output["edge_logits"]).detach().cpu().numpy()
        object_scores = torch.sigmoid(output["object_logits"]).detach().cpu().numpy()
        truth = np.asarray(item["source_particle_id"], dtype=np.int64)
        pred = connected_components_from_edges(
            truth.shape[0],
            np.asarray(item["edge_index"]),
            edge_scores,
            object_scores,
            np.asarray(item["edge_attr"]) if "edge_attr" in item else None,
            np.asarray(item["features"]),
            edge_threshold=edge_threshold,
            object_threshold=object_threshold,
        )
        rows.append(
            grouping_metrics(
                truth,
                pred,
                np.asarray(item["hit_energy"], dtype=np.float32),
                object_scores,
                object_threshold=object_threshold,
            )
        )

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
    *,
    object_threshold: float = 0.5,
) -> dict[str, float]:
    return {
        "ari": adjusted_rand_index(truth, pred),
        "pairwise_f1": pairwise_f1(truth, pred),
        "split_rate": split_rate(truth, pred),
        "merge_rate": merge_rate(truth, pred),
        "object_accuracy": float(np.mean((object_scores >= object_threshold) == (truth >= 0))) if truth.size else 0.0,
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


def model_selection_score(metrics: dict[str, float]) -> float:
    return float(
        metrics.get("ari", 0.0)
        + 0.35 * metrics.get("pairwise_f1", 0.0)
        + 0.15 * metrics.get("object_accuracy", 0.0)
        - 0.35 * metrics.get("split_rate", 0.0)
        - 0.35 * metrics.get("merge_rate", 0.0)
        - 0.10 * metrics.get("energy_error", 0.0)
    )


def is_reasonable_edge_model(metrics: dict[str, float], config: EdgeTrainConfig) -> bool:
    return (
        metrics.get("ari", 0.0) >= config.target_ari
        and metrics.get("pairwise_f1", 0.0) >= config.target_pairwise_f1
        and metrics.get("split_rate", 1.0) <= config.target_split_rate
        and metrics.get("merge_rate", 1.0) <= config.target_merge_rate
        and metrics.get("object_accuracy", 0.0) >= config.target_object_accuracy
        and metrics.get("energy_error", 1.0) <= config.target_energy_error
    )


def sweep_edge_thresholds(
    model: EdgeTrackNetTiny,
    dataset: EdgeWindowDataset,
    csv_path: str | Path,
    *,
    max_windows: int,
    device: str,
    edge_thresholds: Iterable[float] | None = None,
    object_thresholds: Iterable[float] | None = None,
) -> dict[str, object]:
    edge_thresholds = list(edge_thresholds or [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95])
    object_thresholds = list(object_thresholds or [0.20, 0.35, 0.50, 0.65, 0.80, 0.90])
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    for edge_threshold in edge_thresholds:
        for object_threshold in object_thresholds:
            metrics = evaluate_edge_model(
                model,
                dataset,
                max_windows=max_windows,
                device=device,
                edge_threshold=float(edge_threshold),
                object_threshold=float(object_threshold),
            )
            score = model_selection_score(metrics)
            row = {
                "edge_threshold": float(edge_threshold),
                "object_threshold": float(object_threshold),
                "score": score,
                **metrics,
            }
            rows.append(row)
            if best_row is None or score > float(best_row["score"]):
                best_row = row
    write_dict_rows(csv_path, rows)
    return {"enabled": True, "rows": len(rows), "csv": Path(csv_path).as_posix(), "best": best_row or {}}


def evaluate_phase1(
    checkpoint_path: str | Path,
    manifest_paths: list[str | Path],
    output_dir: str | Path,
    *,
    normalization: str | Path | None = None,
    label_sources: list[str] | None = None,
    device: str = "cpu",
    max_windows: int = 256,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = EdgeTrackNetTinyConfig(**checkpoint["model_config"])
    model = EdgeTrackNetTiny(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    edge_threshold = float(checkpoint.get("selected_edge_threshold", 0.5))
    object_threshold = float(checkpoint.get("selected_object_threshold", 0.5))
    normalization_data = resolve_evaluation_normalization(checkpoint, normalization)

    summary_rows: list[dict[str, object]] = []
    bucket_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    pr_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    failure_examples: list[dict[str, object]] = []
    latency_values: list[float] = []

    for manifest_idx, manifest_path in enumerate(manifest_paths):
        manifest_path = Path(manifest_path)
        rows = load_manifest_rows(manifest_path, split="test")
        if label_sources:
            rows = [row for row in rows if row.get("label_source", "teacher_dbscan") in set(label_sources)]
        selected_rows = rows[:max_windows]
        item_results = []
        edge_scores_all: list[np.ndarray] = []
        edge_labels_all: list[np.ndarray] = []
        for row in selected_rows:
            item = load_edge_window_item(row)
            apply_item_normalization(item, normalization_data)
            batch = move_batch_tensors(collate_edge_windows([item]), device)
            start = time.perf_counter()
            with torch.no_grad():
                result = model(batch["features"], batch["edge_index"], batch.get("edge_attr"))
            latency_ms = (time.perf_counter() - start) * 1000.0
            latency_values.append(latency_ms)
            edge_scores = torch.sigmoid(result["edge_logits"]).detach().cpu().numpy()
            object_scores = torch.sigmoid(result["object_logits"]).detach().cpu().numpy()
            truth = np.asarray(item["source_particle_id"], dtype=np.int64)
            pred = connected_components_from_edges(
                truth.shape[0],
                np.asarray(item["edge_index"]),
                edge_scores,
                object_scores,
                np.asarray(item["edge_attr"]) if "edge_attr" in item else None,
                np.asarray(item["features"]),
                edge_threshold=edge_threshold,
                object_threshold=object_threshold,
            )
            metrics = grouping_metrics(
                truth,
                pred,
                np.asarray(item["hit_energy"], dtype=np.float32),
                object_scores,
                object_threshold=object_threshold,
            )
            item_results.append(
                {
                    "row": row,
                    "metrics": metrics,
                    "truth": truth,
                    "pred": pred,
                    "item": item,
                    "edge_scores": edge_scores,
                    "object_scores": object_scores,
                }
            )
            edge_mask = np.asarray(item["edge_label"]) >= 0
            if np.any(edge_mask):
                edge_scores_all.append(edge_scores[edge_mask])
                edge_labels_all.append(np.asarray(item["edge_label"])[edge_mask])
            object_rows.append(
                {
                    "manifest": manifest_path.as_posix(),
                    "path": row["path"],
                    **object_iou_metrics(truth, pred, np.asarray(item["hit_energy"], dtype=np.float32)),
                }
            )

        label_source = selected_rows[0].get("label_source", "teacher_dbscan") if selected_rows else ""
        aggregate = aggregate_metric_rows([item["metrics"] for item in item_results])
        summary_row = {
            "manifest": manifest_path.as_posix(),
            "label_source": label_source,
            "windows": len(item_results),
            "edge_threshold": edge_threshold,
            "object_threshold": object_threshold,
            **aggregate,
        }
        summary_rows.append(summary_row)
        bucket_rows.extend(build_bucket_rows(manifest_path, item_results))
        sweep_rows.extend(threshold_sweep_rows(manifest_path, label_source, item_results))
        if edge_scores_all:
            pr_rows.extend(edge_pr_curve(np.concatenate(edge_scores_all), np.concatenate(edge_labels_all), manifest_path, label_source))
        failure_examples.extend(
            write_failure_examples(
                output / "failure_examples",
                manifest_path,
                item_results[:],
                max_examples=4,
                prefix=f"manifest_{manifest_idx:02d}",
            )
        )

    write_dict_rows(output / "metrics_by_bucket.csv", bucket_rows)
    write_dict_rows(output / "object_iou_metrics.csv", object_rows)
    write_dict_rows(output / "edge_pr_curve.csv", pr_rows)
    write_dict_rows(output / "threshold_sweep.csv", sweep_rows)
    latency = {
        "count": len(latency_values),
        "p50_ms": float(np.percentile(latency_values, 50)) if latency_values else 0.0,
        "p95_ms": float(np.percentile(latency_values, 95)) if latency_values else 0.0,
        "p99_ms": float(np.percentile(latency_values, 99)) if latency_values else 0.0,
        "device": device,
    }
    (output / "latency_cpu.json").write_text(json.dumps(latency, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "checkpoint": Path(checkpoint_path).as_posix(),
        "normalization": normalization_data.get("source", "") if normalization_data else "",
        "manifests": [Path(path).as_posix() for path in manifest_paths],
        "rows": summary_rows,
        "latency": latency,
        "failure_examples": failure_examples,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_edge_window_item(row: dict[str, str]) -> dict[str, np.ndarray | str]:
    with np.load(row["path"], allow_pickle=False) as data:
        item: dict[str, np.ndarray | str] = {
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
        if "raw_features" in data.files:
            item["raw_features"] = data["raw_features"].astype(np.float32)
        if "edge_attr" in data.files:
            item["edge_attr"] = data["edge_attr"].astype(np.float32)
        return item


def resolve_evaluation_normalization(checkpoint: dict[str, object], normalization: str | Path | None) -> dict[str, object] | None:
    if normalization is not None:
        path = Path(normalization)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source"] = path.as_posix()
        return data
    checkpoint_norm = checkpoint.get("normalization")
    if isinstance(checkpoint_norm, dict):
        data = dict(checkpoint_norm)
        data["source"] = str(checkpoint.get("normalization_path", "checkpoint"))
        return data
    return None


def apply_item_normalization(item: dict[str, np.ndarray | str], normalization: dict[str, object] | None) -> None:
    if normalization is None or "raw_features" not in item:
        return
    raw = np.asarray(item["raw_features"], dtype=np.float32)
    mean = np.asarray(normalization.get("mean", []), dtype=np.float32)
    std = np.asarray(normalization.get("std", []), dtype=np.float32)
    if mean.shape[0] != raw.shape[1] or std.shape[0] != raw.shape[1]:
        raise ValueError(
            f"Normalization dimension mismatch: raw_features has {raw.shape[1]} columns, "
            f"normalization has mean={mean.shape[0]} std={std.shape[0]}."
        )
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    item["features"] = ((raw - mean) / std).astype(np.float32)


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "ari": 0.0,
            "pairwise_f1": 0.0,
            "split_rate": 0.0,
            "merge_rate": 0.0,
            "object_accuracy": 0.0,
            "energy_error": 0.0,
        }
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def threshold_sweep_rows(
    manifest_path: Path,
    label_source: str,
    item_results: list[dict[str, object]],
    *,
    edge_thresholds: Iterable[float] | None = None,
    object_thresholds: Iterable[float] | None = None,
) -> list[dict[str, object]]:
    edge_thresholds = list(edge_thresholds or [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95])
    object_thresholds = list(object_thresholds or [0.20, 0.35, 0.50, 0.65, 0.80, 0.90])
    rows: list[dict[str, object]] = []
    for edge_threshold in edge_thresholds:
        for object_threshold in object_thresholds:
            metrics_per_window = []
            for result in item_results:
                item = result["item"]
                truth = np.asarray(result["truth"], dtype=np.int64)
                edge_scores = np.asarray(result["edge_scores"], dtype=np.float32)
                object_scores = np.asarray(result["object_scores"], dtype=np.float32)
                pred = connected_components_from_edges(
                    truth.shape[0],
                    np.asarray(item["edge_index"]),
                    edge_scores,
                    object_scores,
                    np.asarray(item["edge_attr"]) if "edge_attr" in item else None,
                    np.asarray(item["features"]),
                    edge_threshold=float(edge_threshold),
                    object_threshold=float(object_threshold),
                )
                metrics_per_window.append(
                    grouping_metrics(
                        truth,
                        pred,
                        np.asarray(item["hit_energy"], dtype=np.float32),
                        object_scores,
                        object_threshold=float(object_threshold),
                    )
                )
            metrics = aggregate_metric_rows(metrics_per_window)
            rows.append(
                {
                    "manifest": manifest_path.as_posix(),
                    "label_source": label_source,
                    "windows": len(item_results),
                    "edge_threshold": float(edge_threshold),
                    "object_threshold": float(object_threshold),
                    "score": model_selection_score(metrics),
                    **metrics,
                }
            )
    return rows


def build_bucket_rows(manifest_path: Path, item_results: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], list[dict[str, float]]] = {}
    for result in item_results:
        row = result["row"]
        metrics = result["metrics"]
        assert isinstance(row, dict) and isinstance(metrics, dict)
        bucket_values = {
            "hit_count": bucket_numeric(float(row.get("n_hits", 0)), [64, 256, 1024], ["<=64", "65-256", "257-1024", ">1024"]),
            "negative_edges": bucket_numeric(float(row.get("negative_edges", 0)), [0, 32, 256], ["0", "1-32", "33-256", ">256"]),
            "particles": bucket_numeric(float(row.get("particles", 0)), [3, 20, 100], ["<=3", "4-20", "21-100", ">100"]),
        }
        for bucket_name, bucket in bucket_values.items():
            buckets.setdefault((bucket_name, bucket), []).append(metrics)
    out = []
    for (bucket_name, bucket), rows in sorted(buckets.items()):
        out.append({"manifest": manifest_path.as_posix(), "bucket_name": bucket_name, "bucket": bucket, "windows": len(rows), **aggregate_metric_rows(rows)})
    return out


def bucket_numeric(value: float, thresholds: list[float], labels: list[str]) -> str:
    for threshold, label in zip(thresholds, labels, strict=False):
        if value <= threshold:
            return label
    return labels[-1]


def edge_pr_curve(scores: np.ndarray, labels: np.ndarray, manifest_path: Path, label_source: str) -> list[dict[str, object]]:
    rows = []
    labels = labels.astype(np.int8)
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = scores >= threshold
        positive = labels == 1
        negative = labels == 0
        tp = int(np.sum(pred & positive))
        fp = int(np.sum(pred & negative))
        fn = int(np.sum((~pred) & positive))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "manifest": manifest_path.as_posix(),
                "label_source": label_source,
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return rows


def object_iou_metrics(truth: np.ndarray, pred: np.ndarray, energy: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    truth_ids = sorted(set(int(label) for label in truth.tolist()) - {-1})
    pred_ids = sorted(set(int(label) for label in pred.tolist()) - {-1})
    matches = 0
    energy_errors = []
    used_pred: set[int] = set()
    for truth_id in truth_ids:
        truth_mask = truth == truth_id
        best_pred = None
        best_iou = 0.0
        for pred_id in pred_ids:
            if pred_id in used_pred:
                continue
            pred_mask = pred == pred_id
            inter = int(np.sum(truth_mask & pred_mask))
            union = int(np.sum(truth_mask | pred_mask))
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_id
        if best_pred is not None and best_iou >= threshold:
            matches += 1
            used_pred.add(best_pred)
            truth_energy = float(np.sum(energy[truth_mask]))
            pred_energy = float(np.sum(energy[pred == best_pred]))
            energy_errors.append(abs(pred_energy - truth_energy) / max(truth_energy, 1e-6))
    precision = matches / max(len(pred_ids), 1)
    recall = matches / max(len(truth_ids), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "truth_objects": len(truth_ids),
        "pred_objects": len(pred_ids),
        "matched_objects": matches,
        "precision_iou_0_5": float(precision),
        "recall_iou_0_5": float(recall),
        "f1_iou_0_5": float(f1),
        "matched_energy_error": float(np.mean(energy_errors)) if energy_errors else 0.0,
    }


def write_failure_examples(
    output_dir: Path,
    manifest_path: Path,
    item_results: list[dict[str, object]],
    *,
    max_examples: int,
    prefix: str | None = None,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    item_results.sort(key=lambda result: float(result["metrics"]["ari"]))  # type: ignore[index]
    rows = []
    for idx, result in enumerate(item_results[:max_examples]):
        row = result["row"]
        metrics = result["metrics"]
        assert isinstance(row, dict) and isinstance(metrics, dict)
        stem = prefix or manifest_path.stem
        path = output_dir / f"{stem}_{idx:02d}.png"
        try:
            write_failure_plot(path, result)
        except Exception:
            continue
        rows.append({"manifest": manifest_path.as_posix(), "path": row["path"], "plot": path.as_posix(), **metrics})
    return rows


def write_failure_plot(path: Path, result: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    item = result["item"]
    truth = result["truth"]
    pred = result["pred"]
    assert isinstance(item, dict)
    x = np.asarray(item["features"])[:, 0]
    y = np.asarray(item["features"])[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].scatter(x, y, c=truth, s=8, cmap="tab20")
    axes[0].set_title("truth")
    axes[1].scatter(x, y, c=pred, s=8, cmap="tab20")
    axes[1].set_title("prediction")
    for ax in axes:
        ax.set_xlabel("x feature")
        ax.set_ylabel("y feature")
    fig.savefig(path, dpi=140)
    plt.close(fig)


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

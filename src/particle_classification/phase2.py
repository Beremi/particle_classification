from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .data.particles import write_dict_rows
from .data.phase2_dataset import (
    POINT_FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
    Phase2DatasetConfig,
    build_phase2_dataset,
    load_phase2_manifest,
)


@dataclass(frozen=True)
class Phase2ModelConfig:
    backbone: str = "deepsets"
    objective: str = "ae"
    point_dim: int = len(POINT_FEATURE_NAMES)
    summary_dim: int = len(SUMMARY_FEATURE_NAMES)
    hidden_dim: int = 128
    latent_dim: int = 64
    decoder_points: int = 64
    dropout: float = 0.05
    transformer_heads: int = 4
    edgeconv_k: int = 12
    n_clusters: int = 16


@dataclass(frozen=True)
class Phase2TrainConfig:
    seed: int = 20260503
    budget: str = "smoke"
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_train_items: int | None = None
    max_eval_items: int = 2048
    steps: int | None = None
    patience: int = 8
    min_steps: int = 50
    eval_interval: int = 50
    amp: bool = True
    grad_clip: float = 1.0
    repeat_top_seeds: int = 2
    run_limit: int | None = None


class Phase2ParticleDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, *, split: str = "train", max_items: int | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = self.dataset_dir / "manifest.csv"
        self.rows = load_phase2_manifest(self.manifest_path, split=split)
        if max_items is not None:
            self.rows = self.rows[:max_items]
        self.normalization = json.loads((self.dataset_dir / "normalization.json").read_text(encoding="utf-8"))
        self._chunk_path: str | None = None
        self._chunk: dict[str, np.ndarray] | None = None
        self.indices_by_bucket: dict[str, list[int]] = {}
        for idx, row in enumerate(self.rows):
            self.indices_by_bucket.setdefault(row.get("size_bucket", "unknown"), []).append(idx)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        data = self._load_chunk(row["chunk_path"])
        chunk_row = int(row["chunk_row"])
        start = int(data["offsets"][chunk_row])
        end = int(data["offsets"][chunk_row + 1])
        points = data["points"][start:end].astype(np.float32)
        summary = data["summary"][chunk_row].astype(np.float32)
        points = normalize_array(points, self.normalization["point_mean"], self.normalization["point_std"])
        summary = normalize_array(summary, self.normalization["summary_mean"], self.normalization["summary_std"])
        return {
            "points": points.astype(np.float32),
            "summary": summary.astype(np.float32),
            "row": row,
            "n_hits": int(float(row.get("n_hits", 0))),
            "view_n_hits": int(float(row.get("view_n_hits", points.shape[0]))),
            "size_bucket": row.get("size_bucket", "unknown"),
        }

    def _load_chunk(self, chunk_path: str) -> dict[str, np.ndarray]:
        if self._chunk_path == chunk_path and self._chunk is not None:
            return self._chunk
        with np.load(chunk_path, allow_pickle=False) as data:
            self._chunk = {key: data[key] for key in data.files}
        self._chunk_path = chunk_path
        return self._chunk


def normalize_array(values: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    mean_arr = np.asarray(list(mean), dtype=np.float32)
    std_arr = np.asarray(list(std), dtype=np.float32)
    return (values - mean_arr) / np.maximum(std_arr, 1e-6)


def random_phase2_batch(dataset: Phase2ParticleDataset, batch_size: int) -> list[dict[str, object]]:
    groups = {key: value for key, value in dataset.indices_by_bucket.items() if value}
    if len(groups) >= 2 and batch_size > 1:
        buckets = sorted(groups)
        selected: list[int] = []
        base = batch_size // len(buckets)
        remainder = batch_size % len(buckets)
        for bucket_idx, bucket in enumerate(buckets):
            draws = base + (1 if bucket_idx < remainder else 0)
            selected.extend(random.choice(groups[bucket]) for _ in range(draws))
        random.shuffle(selected)
        return [dataset[idx] for idx in selected]
    if len(dataset) <= batch_size:
        return [dataset[idx] for idx in range(len(dataset))]
    return [dataset[idx] for idx in random.sample(range(len(dataset)), k=batch_size)]


def collate_phase2(items: list[dict[str, object]]) -> dict[str, object]:
    max_points = max(int(np.asarray(item["points"]).shape[0]) for item in items)
    point_dim = int(np.asarray(items[0]["points"]).shape[1])
    points = np.zeros((len(items), max_points, point_dim), dtype=np.float32)
    mask = np.zeros((len(items), max_points), dtype=bool)
    for idx, item in enumerate(items):
        arr = np.asarray(item["points"], dtype=np.float32)
        points[idx, : arr.shape[0]] = arr
        mask[idx, : arr.shape[0]] = True
    return {
        "points": torch.from_numpy(points),
        "mask": torch.from_numpy(mask),
        "summary": torch.from_numpy(np.stack([np.asarray(item["summary"], dtype=np.float32) for item in items], axis=0)),
        "rows": [item["row"] for item in items],
        "size_bucket": [str(item["size_bucket"]) for item in items],
        "n_hits": np.asarray([int(item["n_hits"]) for item in items], dtype=np.int32),
    }


class ParticleEncoder(nn.Module):
    def __init__(self, config: Phase2ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.hit_mlp = nn.Sequential(
            nn.Linear(config.point_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )
        self.summary_mlp = nn.Sequential(
            nn.Linear(config.summary_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.attention = nn.Linear(h, 1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=max(1, config.transformer_heads),
            dim_feedforward=h * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.position_mlp = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h))
        self.out = nn.Sequential(
            nn.Linear(h * 4, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.latent_dim),
        )
        self.apply(init_phase2_weights)

    def forward(self, points: torch.Tensor, mask: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        h = self.hit_mlp(points)
        if self.config.backbone == "edgeconv":
            h = h + self.edgeconv(points, h, mask)
        elif self.config.backbone == "settransformer":
            h = self.transformer(h, src_key_padding_mask=~mask)
        elif self.config.backbone == "pointtransformer":
            h = self.transformer(h + self.position_mlp(points[:, :, :3]), src_key_padding_mask=~mask)
        elif self.config.backbone != "deepsets":
            raise ValueError(f"Unknown Phase 2 backbone: {self.config.backbone}")

        pooled = self.pool(h, mask)
        summary_h = self.summary_mlp(summary)
        return self.out(torch.cat([pooled, summary_h], dim=-1))

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=h.dtype)
        denom = torch.clamp(mask_f.sum(dim=1), min=1.0)
        mean_pool = (h * mask_f).sum(dim=1) / denom
        fill_value = -1e4
        max_pool = h.masked_fill(~mask.unsqueeze(-1), fill_value).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        logits = self.attention(h).squeeze(-1).masked_fill(~mask, fill_value)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        att_pool = (h * weights).sum(dim=1)
        return torch.cat([max_pool, mean_pool, att_pool], dim=-1)

    def edgeconv(self, points: torch.Tensor, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, n_points, hidden = h.shape
        if n_points <= 1:
            return torch.zeros_like(h)
        coords = points[:, :, :3]
        dist = torch.cdist(coords, coords)
        invalid = ~mask
        dist = dist.masked_fill(invalid[:, None, :], torch.inf)
        dist = dist.masked_fill(torch.eye(n_points, dtype=torch.bool, device=points.device).unsqueeze(0), torch.inf)
        k = min(max(1, self.config.edgeconv_k), n_points - 1)
        neighbors = torch.topk(dist, k=k, dim=-1, largest=False).indices
        expanded = neighbors.unsqueeze(-1).expand(batch, n_points, k, hidden)
        h_neighbors = torch.gather(h.unsqueeze(1).expand(batch, n_points, n_points, hidden), 2, expanded)
        h_center = h.unsqueeze(2).expand_as(h_neighbors)
        messages = self.edge_mlp(torch.cat([h_center, h_neighbors - h_center], dim=-1))
        messages = messages.masked_fill(~mask[:, :, None, None], 0.0)
        return messages.max(dim=2).values


class Phase2ParticleModel(nn.Module):
    def __init__(self, config: Phase2ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ParticleEncoder(config)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.decoder_points * config.point_dim),
        )
        self.dec_head = nn.Linear(config.latent_dim, config.n_clusters)
        self.mu_head = nn.Linear(config.latent_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.latent_dim, config.latent_dim)
        self.apply(init_phase2_weights)

    def forward(self, points: torch.Tensor, mask: torch.Tensor, summary: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(points, mask, summary)
        decoded = self.decoder(z).view(points.shape[0], self.config.decoder_points, self.config.point_dim)
        return {
            "z": torch.nn.functional.normalize(z, dim=-1),
            "decoded": decoded,
            "cluster_logits": self.dec_head(z),
            "mu": self.mu_head(z),
            "logvar": torch.clamp(self.logvar_head(z), min=-8.0, max=8.0),
        }


def init_phase2_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def phase2_loss(
    model: Phase2ParticleModel,
    batch: dict[str, object],
    *,
    objective: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    points = batch["points"]
    mask = batch["mask"]
    summary = batch["summary"]
    if objective == "denoising_ae":
        noisy = augment_points(points, mask, jitter=0.03, dropout=0.08)
        output = model(noisy, mask, summary)
        loss = masked_chamfer_loss(output["decoded"], points, mask)
        return loss, {"loss": float(loss.detach().cpu()), "reconstruction": float(loss.detach().cpu())}
    if objective == "masked_ae":
        masked_points, masked_mask = random_point_mask(points, mask, keep_fraction=0.75)
        output = model(masked_points, masked_mask, summary)
        loss = masked_chamfer_loss(output["decoded"], points, mask)
        return loss, {"loss": float(loss.detach().cpu()), "reconstruction": float(loss.detach().cpu())}
    if objective == "contrastive":
        view_a = augment_points(points, mask, jitter=0.02, dropout=0.05)
        view_b = augment_points(points, mask, jitter=0.02, dropout=0.05)
        z_a = model(view_a, mask, summary)["z"]
        z_b = model(view_b, mask, summary)["z"]
        loss = info_nce_loss(z_a, z_b)
        align = torch.mean(torch.linalg.norm(z_a - z_b, dim=-1))
        return loss, {
            "loss": float(loss.detach().cpu()),
            "contrastive": float(loss.detach().cpu()),
            "alignment": float(align.detach().cpu()),
        }
    if objective == "dec":
        output = model(points, mask, summary)
        probs = torch.softmax(output["cluster_logits"], dim=-1)
        target = dec_target_distribution(probs.detach())
        loss = torch.nn.functional.kl_div(torch.log(torch.clamp(probs, min=1e-8)), target, reduction="batchmean")
        entropy = torch.mean(torch.sum(-probs * torch.log(torch.clamp(probs, min=1e-8)), dim=-1))
        return loss, {"loss": float(loss.detach().cpu()), "cluster_entropy": float(entropy.detach().cpu())}
    if objective == "vade":
        output = model(points, mask, summary)
        std = torch.exp(0.5 * output["logvar"])
        z_sample = output["mu"] + torch.randn_like(std) * std
        decoded = model.decoder(z_sample).view(points.shape[0], model.config.decoder_points, model.config.point_dim)
        recon = masked_chamfer_loss(decoded, points, mask)
        kl = -0.5 * torch.mean(1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp())
        logits = output["cluster_logits"]
        probs = torch.softmax(logits, dim=-1)
        entropy = torch.mean(torch.sum(-probs * torch.log(torch.clamp(probs, min=1e-8)), dim=-1))
        loss = recon + 0.05 * kl - 0.001 * entropy
        return loss, {
            "loss": float(loss.detach().cpu()),
            "reconstruction": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "cluster_entropy": float(entropy.detach().cpu()),
        }
    if objective not in {"ae", "point_ae"}:
        raise ValueError(f"Unknown Phase 2 objective: {objective}")
    output = model(points, mask, summary)
    loss = masked_chamfer_loss(output["decoded"], points, mask)
    return loss, {"loss": float(loss.detach().cpu()), "reconstruction": float(loss.detach().cpu())}


def masked_chamfer_loss(decoded: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = target[:, :, : decoded.shape[-1]]
    losses = []
    for pred_item, target_item, mask_item in zip(decoded, target, mask, strict=True):
        active = target_item[mask_item]
        if active.numel() == 0:
            continue
        dist = torch.cdist(pred_item[:, :5], active[:, :5])
        losses.append(dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())
    if not losses:
        return decoded.sum() * 0.0
    return torch.stack(losses).mean()


def augment_points(points: torch.Tensor, mask: torch.Tensor, *, jitter: float, dropout: float) -> torch.Tensor:
    noise = torch.randn_like(points) * jitter
    out = points + noise * mask.unsqueeze(-1).to(points.dtype)
    if dropout > 0.0:
        keep = (torch.rand(mask.shape, device=points.device) > dropout) | ~mask
        out = torch.where(keep.unsqueeze(-1), out, torch.zeros_like(out))
    # Mild time stretch on normalized/scaled time channels.
    stretch = torch.empty((points.shape[0], 1, 1), device=points.device, dtype=points.dtype).uniform_(0.95, 1.05)
    out[:, :, 2:4] = out[:, :, 2:4] * stretch
    return out


def random_point_mask(points: torch.Tensor, mask: torch.Tensor, *, keep_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    keep = (torch.rand(mask.shape, device=points.device) < keep_fraction) & mask
    ensure = mask.any(dim=1)
    if bool(torch.any(ensure & ~keep.any(dim=1))):
        first = torch.argmax(mask.to(torch.int64), dim=1)
        keep[torch.arange(mask.shape[0], device=points.device), first] |= ensure
    return torch.where(keep.unsqueeze(-1), points, torch.zeros_like(points)), keep


def info_nce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z_a = torch.nn.functional.normalize(z_a, dim=-1)
    z_b = torch.nn.functional.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.T, labels)
    )


def dec_target_distribution(probs: torch.Tensor) -> torch.Tensor:
    weight = probs.pow(2) / torch.clamp(probs.sum(dim=0, keepdim=True), min=1e-8)
    return weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1e-8)


def phase2_run_grid(budget: str) -> list[Phase2ModelConfig]:
    backbones = ["deepsets", "edgeconv", "settransformer", "pointtransformer"]
    objectives = ["ae", "denoising_ae", "masked_ae", "contrastive", "dec", "vade"]
    if budget == "smoke":
        return [
            Phase2ModelConfig(backbone="deepsets", objective="ae", hidden_dim=48, latent_dim=24, decoder_points=16),
            Phase2ModelConfig(backbone="edgeconv", objective="contrastive", hidden_dim=48, latent_dim=24, decoder_points=16, edgeconv_k=4),
        ]
    configs = []
    for backbone in backbones:
        for objective in objectives:
            configs.append(Phase2ModelConfig(backbone=backbone, objective=objective))
    return configs


def budget_defaults(config: Phase2TrainConfig) -> dict[str, int | None]:
    if config.budget == "smoke":
        return {"steps": config.steps or 8, "max_train_items": config.max_train_items or 128, "max_eval_items": min(config.max_eval_items, 64)}
    if config.budget == "fast":
        return {"steps": config.steps or 500, "max_train_items": config.max_train_items or 20_000, "max_eval_items": min(config.max_eval_items, 2048)}
    if config.budget == "overnight":
        return {"steps": config.steps or 5_000, "max_train_items": config.max_train_items, "max_eval_items": config.max_eval_items}
    raise ValueError(f"Unknown Phase 2 budget: {config.budget}")


def train_phase2_sweep(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    config: Phase2TrainConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, object]:
    config = config or Phase2TrainConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(output_dir)
    runs_dir = output / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    defaults = budget_defaults(config)
    train_dataset = Phase2ParticleDataset(dataset_dir, split="train", max_items=defaults["max_train_items"])
    val_dataset = Phase2ParticleDataset(dataset_dir, split="val", max_items=defaults["max_eval_items"])
    if not len(train_dataset):
        raise RuntimeError("No Phase 2 train particles found.")
    if not len(val_dataset):
        val_dataset = train_dataset
    normalization = train_dataset.normalization

    run_configs = phase2_run_grid(config.budget)
    if config.run_limit is not None:
        run_configs = run_configs[: config.run_limit]
    rows = []
    for run_idx, model_config in enumerate(run_configs, start=1):
        model_config = Phase2ModelConfig(
            backbone=model_config.backbone,
            objective=model_config.objective,
            point_dim=int(normalization["point_dim"]),
            summary_dim=int(normalization["summary_dim"]),
            hidden_dim=model_config.hidden_dim,
            latent_dim=model_config.latent_dim,
            decoder_points=model_config.decoder_points,
            dropout=model_config.dropout,
            transformer_heads=model_config.transformer_heads,
            edgeconv_k=model_config.edgeconv_k,
            n_clusters=model_config.n_clusters,
        )
        run_name = f"{run_idx:02d}_{model_config.backbone}_{model_config.objective}"
        if verbose:
            print(f"[{run_idx}/{len(run_configs)}] {run_name}", flush=True)
        row = train_phase2_run(
            train_dataset,
            val_dataset,
            runs_dir / run_name,
            model_config=model_config,
            train_config=config,
            steps=int(defaults["steps"]),
            device=device,
            verbose=verbose,
        )
        rows.append(row)
        write_dict_rows(output / "sweep_summary.csv", rows)

    best = max(rows, key=lambda item: float(item.get("selection_score", float("-inf")))) if rows else {}
    summary = {
        "dataset": Path(dataset_dir).as_posix(),
        "output": output.as_posix(),
        "budget": config.budget,
        "runs": rows,
        "best_run": best,
        "train_items": len(train_dataset),
        "val_items": len(val_dataset),
    }
    (output / "sweep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def train_phase2_run(
    train_dataset: Phase2ParticleDataset,
    val_dataset: Phase2ParticleDataset,
    output_dir: Path,
    *,
    model_config: Phase2ModelConfig,
    train_config: Phase2TrainConfig,
    steps: int,
    device: str,
    verbose: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = Phase2ParticleModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=train_config.learning_rate * 0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    best_loss = float("inf")
    best_state = None
    best_step = 0
    bad_evals = 0
    rows = []
    start_time = time.perf_counter()
    for step in range(1, steps + 1):
        batch = move_phase2_batch(collate_phase2(random_phase2_batch(train_dataset, train_config.batch_size)), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available())):
            loss, loss_metrics = phase2_loss(model, batch, objective=model_config.objective)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % train_config.eval_interval == 0 or step == steps:
            val = evaluate_phase2_model(model, val_dataset, objective=model_config.objective, max_items=train_config.max_eval_items, device=device)
            row = {
                "step": step,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **loss_metrics,
                **{f"val_{key}": value for key, value in val.items()},
            }
            rows.append(row)
            write_dict_rows(output_dir / "metrics.csv", rows)
            if verbose:
                print(
                    f"  step {step}/{steps}: loss={row['loss']:.4f}, val_loss={row['val_loss']:.4f}, lr={row['lr']:.2e}",
                    flush=True,
                )
            if val["loss"] < best_loss:
                best_loss = float(val["loss"])
                best_step = step
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                bad_evals = 0
            else:
                bad_evals += 1
            if step >= train_config.min_steps and bad_evals >= train_config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "best_step": best_step,
            "best_val_loss": best_loss,
        },
        checkpoint_path,
    )
    embeddings_path = output_dir / "val_embeddings.npz"
    embedding_summary = write_phase2_embeddings(model, val_dataset, embeddings_path, max_items=train_config.max_eval_items, device=device)
    duration = time.perf_counter() - start_time
    selection_score = -best_loss + 0.02 * float(embedding_summary.get("embedding_std", 0.0))
    summary = {
        "run": output_dir.name,
        "backbone": model_config.backbone,
        "objective": model_config.objective,
        "checkpoint": checkpoint_path.as_posix(),
        "metrics": (output_dir / "metrics.csv").as_posix(),
        "embeddings": embeddings_path.as_posix(),
        "best_step": best_step,
        "best_val_loss": best_loss,
        "selection_score": selection_score,
        "duration_s": duration,
        **embedding_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


@torch.no_grad()
def evaluate_phase2_model(
    model: Phase2ParticleModel,
    dataset: Phase2ParticleDataset,
    *,
    objective: str,
    max_items: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    if not len(dataset):
        return {"loss": 0.0}
    indices = np.linspace(0, len(dataset) - 1, num=min(len(dataset), max_items), dtype=int)
    losses = []
    batch_size = 64
    for start in range(0, len(indices), batch_size):
        items = [dataset[int(idx)] for idx in indices[start : start + batch_size].tolist()]
        batch = move_phase2_batch(collate_phase2(items), device)
        loss, _ = phase2_loss(model, batch, objective=objective)
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses)) if losses else 0.0}


def move_phase2_batch(batch: dict[str, object], device: str) -> dict[str, object]:
    out = dict(batch)
    for key in ["points", "mask", "summary"]:
        out[key] = out[key].to(device)
    return out


@torch.no_grad()
def write_phase2_embeddings(
    model: Phase2ParticleModel,
    dataset: Phase2ParticleDataset,
    path: str | Path,
    *,
    max_items: int,
    device: str,
) -> dict[str, object]:
    model.eval()
    indices = np.arange(min(len(dataset), max_items), dtype=int)
    embeddings = []
    row_refs = []
    batch_size = 128
    for start in range(0, len(indices), batch_size):
        items = [dataset[int(idx)] for idx in indices[start : start + batch_size].tolist()]
        batch = move_phase2_batch(collate_phase2(items), device)
        z = model(batch["points"], batch["mask"], batch["summary"])["z"].detach().cpu().numpy()
        embeddings.append(z)
        row_refs.extend(batch["rows"])
    embedding_arr = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, model.config.latent_dim), dtype=np.float32)
    np.savez_compressed(
        path,
        embedding=embedding_arr.astype(np.float32),
        source_path=np.asarray([row["source_path"] for row in row_refs]),
        particle_id=np.asarray([int(float(row["particle_id"])) for row in row_refs], dtype=np.int32),
        n_hits=np.asarray([int(float(row["n_hits"])) for row in row_refs], dtype=np.int32),
        size_bucket=np.asarray([row["size_bucket"] for row in row_refs]),
    )
    return {
        "embedding_count": int(embedding_arr.shape[0]),
        "embedding_dim": int(embedding_arr.shape[1]) if embedding_arr.ndim == 2 else 0,
        "embedding_std": float(np.mean(np.std(embedding_arr, axis=0))) if embedding_arr.size else 0.0,
    }


def evaluate_phase2_experiment(
    dataset_dir: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    max_items: int = 4096,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment = Path(experiment_dir)
    run_summaries = load_run_summaries(experiment)
    rows = []
    cluster_rows = []
    prototype_rows = []
    for run in run_summaries:
        embedding_path = Path(str(run.get("embeddings", "")))
        if not embedding_path.exists():
            continue
        with np.load(embedding_path, allow_pickle=False) as data:
            emb = data["embedding"].astype(np.float32)
            n_hits = data["n_hits"].astype(np.int32) if "n_hits" in data.files else np.zeros(emb.shape[0], dtype=np.int32)
            source_path = data["source_path"] if "source_path" in data.files else np.asarray([""] * emb.shape[0])
            particle_id = data["particle_id"] if "particle_id" in data.files else np.arange(emb.shape[0])
        labels_by_method = cluster_embeddings(emb)
        for method, labels in labels_by_method.items():
            metrics = clustering_metrics(emb, labels)
            rows.append({"run": run["run"], "backbone": run["backbone"], "objective": run["objective"], "cluster_method": method, **metrics})
            for cluster_id, count in cluster_counts(labels).items():
                cluster_rows.append({"run": run["run"], "cluster_method": method, "cluster_id": cluster_id, "count": count})
            prototype_rows.extend(select_prototypes(run["run"], method, emb, labels, source_path, particle_id, n_hits))

    write_dict_rows(output / "cluster_metrics.csv", rows)
    write_dict_rows(output / "cluster_sizes.csv", cluster_rows)
    write_dict_rows(output / "prototypes.csv", prototype_rows)
    summary = {"experiment": experiment.as_posix(), "dataset": Path(dataset_dir).as_posix(), "runs": len(run_summaries), "evaluated_rows": len(rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_run_summaries(experiment_dir: str | Path) -> list[dict[str, object]]:
    path = Path(experiment_dir) / "sweep_summary.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("runs", []))
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(experiment_dir).glob("runs/*/summary.json"))]


def cluster_embeddings(embeddings: np.ndarray) -> dict[str, np.ndarray]:
    if embeddings.shape[0] == 0:
        return {}
    out: dict[str, np.ndarray] = {}
    if 2 <= embeddings.shape[0] < 8:
        out[f"kmeans_{embeddings.shape[0]}"] = kmeans_labels(embeddings, k=embeddings.shape[0], seed=20260503)
        return out
    has_sklearn = importlib.util.find_spec("sklearn") is not None
    has_hdbscan = importlib.util.find_spec("hdbscan") is not None
    for k in [8, 12, 16, 24]:
        if embeddings.shape[0] >= k:
            out[f"kmeans_{k}"] = kmeans_labels(embeddings, k=k, seed=20260503)
            gmm = gmm_labels(embeddings, k=k, seed=20260503)
            if gmm is not None:
                out[f"gmm_{k}"] = gmm
            elif not has_sklearn:
                out[f"gmm_{k}_unavailable_kmeans_fallback"] = kmeans_labels(embeddings, k=k, seed=20260503)
    hdbscan_labels = hdbscan_or_fallback(embeddings)
    if hdbscan_labels is not None:
        out["hdbscan" if has_hdbscan else "hdbscan_unavailable_kmeans_fallback"] = hdbscan_labels
    return out


def kmeans_labels(embeddings: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans  # type: ignore

        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings).astype(np.int32)
    except Exception:
        rng = np.random.default_rng(seed)
        centers = embeddings[rng.choice(embeddings.shape[0], size=k, replace=False)].copy()
        labels = np.zeros(embeddings.shape[0], dtype=np.int32)
        for _ in range(20):
            dist = np.sum((embeddings[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dist, axis=1).astype(np.int32)
            for idx in range(k):
                mask = labels == idx
                if np.any(mask):
                    centers[idx] = np.mean(embeddings[mask], axis=0)
        return labels


def gmm_labels(embeddings: np.ndarray, *, k: int, seed: int) -> np.ndarray | None:
    try:
        from sklearn.mixture import GaussianMixture  # type: ignore

        return GaussianMixture(n_components=k, covariance_type="diag", random_state=seed).fit_predict(embeddings).astype(np.int32)
    except Exception:
        return None


def hdbscan_or_fallback(embeddings: np.ndarray) -> np.ndarray | None:
    if embeddings.shape[0] < 8:
        return None
    try:
        import hdbscan  # type: ignore

        return hdbscan.HDBSCAN(min_cluster_size=max(8, embeddings.shape[0] // 80), min_samples=5).fit_predict(embeddings).astype(np.int32)
    except Exception:
        return kmeans_labels(embeddings, k=min(8, embeddings.shape[0]), seed=20260503)


def clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int32)
    valid = labels >= 0
    n_clusters = len(set(int(label) for label in labels[valid].tolist()))
    metrics: dict[str, float | int] = {
        "n_items": int(labels.shape[0]),
        "n_clusters": int(n_clusters),
        "noise_fraction": float(np.mean(labels < 0)) if labels.size else 0.0,
    }
    if n_clusters >= 2 and int(np.sum(valid)) > n_clusters:
        try:
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score  # type: ignore

            metrics["silhouette"] = float(silhouette_score(embeddings[valid], labels[valid]))
            metrics["davies_bouldin"] = float(davies_bouldin_score(embeddings[valid], labels[valid]))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(embeddings[valid], labels[valid]))
        except Exception:
            metrics["silhouette"] = float(fallback_silhouette(embeddings[valid], labels[valid]))
            metrics["davies_bouldin"] = 0.0
            metrics["calinski_harabasz"] = 0.0
    else:
        metrics["silhouette"] = 0.0
        metrics["davies_bouldin"] = 0.0
        metrics["calinski_harabasz"] = 0.0
    return metrics


def fallback_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if embeddings.shape[0] > 1024:
        embeddings = embeddings[:1024]
        labels = labels[:1024]
    dist = np.sqrt(np.maximum(np.sum((embeddings[:, None, :] - embeddings[None, :, :]) ** 2, axis=2), 0.0))
    values = []
    for idx, label in enumerate(labels.tolist()):
        same = labels == label
        other_labels = [item for item in sorted(set(labels.tolist())) if item != label]
        if np.sum(same) <= 1 or not other_labels:
            continue
        a = float(np.mean(dist[idx, same & (np.arange(labels.shape[0]) != idx)]))
        b = min(float(np.mean(dist[idx, labels == other])) for other in other_labels)
        values.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(values)) if values else 0.0


def cluster_counts(labels: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for label in labels.tolist():
        label = int(label)
        out[label] = out.get(label, 0) + 1
    return out


def select_prototypes(
    run_name: str,
    method: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    source_path: np.ndarray,
    particle_id: np.ndarray,
    n_hits: np.ndarray,
    *,
    max_clusters: int = 32,
) -> list[dict[str, object]]:
    rows = []
    for cluster_id in sorted(set(int(label) for label in labels.tolist()) - {-1})[:max_clusters]:
        indices = np.flatnonzero(labels == cluster_id)
        if indices.size == 0:
            continue
        center = np.mean(embeddings[indices], axis=0)
        best = int(indices[np.argmin(np.sum((embeddings[indices] - center) ** 2, axis=1))])
        rows.append(
            {
                "run": run_name,
                "cluster_method": method,
                "cluster_id": cluster_id,
                "prototype_index": best,
                "source_path": str(source_path[best]),
                "particle_id": int(particle_id[best]),
                "n_hits": int(n_hits[best]),
                "cluster_size": int(indices.size),
            }
        )
    return rows


def generate_phase2_report(
    experiment_dir: str | Path,
    output_path: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    evaluation_dir: str | Path | None = None,
) -> dict[str, object]:
    output = Path(output_path)
    assets = output.parent / "assets" / "phase2_particle_embedding"
    assets.mkdir(parents=True, exist_ok=True)
    experiment = Path(experiment_dir)
    dataset = Path(dataset_dir) if dataset_dir is not None else None
    evaluation = Path(evaluation_dir) if evaluation_dir is not None else experiment / "evaluation"
    run_summaries = load_run_summaries(experiment)
    experiment_summary = load_json(experiment / "sweep_summary.json") if (experiment / "sweep_summary.json").exists() else {}
    dataset_summary = load_json(dataset / "summary.json") if dataset is not None and (dataset / "summary.json").exists() else {}
    cluster_metrics = read_csv_dicts(evaluation / "cluster_metrics.csv") if (evaluation / "cluster_metrics.csv").exists() else []

    plot_paths = make_phase2_plots(experiment, evaluation, assets)
    plot_paths = {key: Path(path).relative_to(output.parent).as_posix() for key, path in plot_paths.items()}
    report = render_phase2_report(dataset_summary, experiment_summary, run_summaries, cluster_metrics, plot_paths)
    output.write_text(report, encoding="utf-8")
    return {"report": output.as_posix(), "assets": assets.as_posix(), "runs": len(run_summaries), "cluster_rows": len(cluster_metrics)}


def make_phase2_plots(experiment: Path, evaluation: Path, assets: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return paths

    for summary_path in sorted(experiment.glob("runs/*/metrics.csv"))[:8]:
        rows = read_csv_dicts(summary_path)
        if not rows:
            continue
        fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
        step = [float(row["step"]) for row in rows]
        loss = [float(row.get("loss", 0.0)) for row in rows]
        val = [float(row.get("val_loss", 0.0)) for row in rows]
        lr = [float(row.get("lr", 0.0)) for row in rows]
        ax1.plot(step, loss, label="train loss")
        ax1.plot(step, val, label="val loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(step, lr, color="tab:green", alpha=0.55, label="LR")
        ax2.set_ylabel("learning rate")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        rel = assets / f"{summary_path.parent.name}_convergence.png"
        fig.savefig(rel, dpi=150)
        plt.close(fig)
        paths[f"{summary_path.parent.name} convergence"] = rel.as_posix()

    metrics = read_csv_dicts(evaluation / "cluster_metrics.csv") if (evaluation / "cluster_metrics.csv").exists() else []
    if metrics:
        labels = [f"{row['run']}\n{row['cluster_method']}" for row in metrics[:20]]
        silhouettes = [float(row.get("silhouette", 0.0)) for row in metrics[:20]]
        fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.45), 4.0))
        ax.bar(np.arange(len(labels)), silhouettes)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
        ax.set_ylabel("silhouette")
        fig.tight_layout()
        rel = assets / "cluster_quality.png"
        fig.savefig(rel, dpi=150)
        plt.close(fig)
        paths["cluster_quality"] = rel.as_posix()
    return paths


def render_phase2_report(
    dataset_summary: dict[str, object],
    experiment_summary: dict[str, object],
    run_summaries: list[dict[str, object]],
    cluster_metrics: list[dict[str, str]],
    plot_paths: dict[str, str],
) -> str:
    budget = experiment_summary.get("budget", "unknown")
    lines = [
        "# Phase 2 Particle Embedding Report",
        "",
        "## Scope",
        "",
        "Phase 2 starts after the active `native-grid-dbscan` separator has produced variable-size particles. The neural networks in this phase do not replace the separator; they learn embeddings for already separated `(x, y, time, energy)` hit sequences, then cluster those embeddings to discover particle families.",
        "",
        "Discovered cluster IDs are morphology groups, not physical particle labels. They must be named later by inspection, simulation truth, or external labels.",
        "",
        "## Current Run Status",
        "",
        f"The tracked metrics in this report come from the completed `{budget}` run artifacts in `local_data/experiments/phase2_particle_sweep_v001`. A smoke run proves the full data/model/evaluation/report path; it is not a final overnight-quality physics result.",
        "",
        "Full overnight command prepared by this implementation:",
        "",
        "```bash",
        "particle-train-phase2-sweep \\",
        "  --dataset local_data/processed/phase2_particles_v001 \\",
        "  --out local_data/experiments/phase2_particle_sweep_v001_overnight \\",
        "  --budget overnight \\",
        "  --device cuda \\",
        "  --batch-size 64",
        "",
        "particle-evaluate-phase2 \\",
        "  --dataset local_data/processed/phase2_particles_v001 \\",
        "  --experiment local_data/experiments/phase2_particle_sweep_v001_overnight \\",
        "  --out local_data/experiments/phase2_particle_sweep_v001_overnight/evaluation \\",
        "  --device cuda",
        "```",
        "",
        "## Data Flow",
        "",
        "Raw `.t3pa` files are separated by the native DBSCAN baseline into NPZ particle shards. `particle-build-phase2-dataset` converts every non-noise particle into one or more variable-hit views, with source-grouped train/validation/test splits by raw file. Large particles are sampled into multiple views capped at `max_points` while keeping full-particle summary descriptors.",
        "",
        f"- dataset views: {dataset_summary.get('ok_views', 'unknown')}",
        f"- particles represented: {dataset_summary.get('particles', 'unknown')}",
        f"- total view hits: {dataset_summary.get('total_view_hits', 'unknown')}",
        f"- split counts: `{dataset_summary.get('splits', {})}`",
        f"- size buckets: `{dataset_summary.get('size_buckets', {})}`",
        "",
        "## Input Tensors",
        "",
        "Per-hit point features: " + ", ".join(f"`{name}`" for name in POINT_FEATURE_NAMES) + ".",
        "",
        "Global summary features are concatenated after point pooling: " + ", ".join(f"`{name}`" for name in SUMMARY_FEATURE_NAMES) + ". Train-split mean/std statistics are stored in `normalization.json` and reused during training and evaluation.",
        "",
        "## Model Family",
        "",
        "The sweep supports four encoder backbones: `DeepSets/PointNet`, `EdgeConv/DGCNN-lite`, `SetTransformer-lite`, and `PointTransformer-lite`. Objectives cover point autoencoding, denoising autoencoding, masked point autoencoding, contrastive ParticleEmbed, DEC, and VaDE-style variational clustering. All models output a fixed-size latent vector `z` per particle view.",
        "",
        "Training uses AdamW, cosine learning-rate scheduling, gradient clipping, automatic mixed precision on CUDA, hit dropout/jitter/time stretch augmentations where relevant, and stratified batches across hit-count buckets.",
        "",
        "| Component | Implementation Detail |",
        "|---|---|",
        "| DeepSets/PointNet | shared hit MLP, max/mean/attention pooling, summary MLP, latent projection |",
        "| EdgeConv/DGCNN-lite | shared hit MLP plus local kNN message block before pooling |",
        "| SetTransformer-lite | two masked self-attention encoder layers over hit tokens |",
        "| PointTransformer-lite | position-conditioned masked transformer over hit tokens |",
        "| AE / denoising / masked AE | fixed-size point decoder trained with Chamfer-style reconstruction loss |",
        "| Contrastive ParticleEmbed | two augmented views with InfoNCE alignment/uniformity objective |",
        "| DEC | cluster-logit KL self-training target for frozen/fine-tuned embeddings |",
        "| VaDE/GMM-VAE | variational latent with reconstruction, KL, and soft mixture entropy terms |",
        "",
        "## Sweep Results",
        "",
        "| Run | Backbone | Objective | Best Step | Val Loss | Embeddings | Selection |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for run in run_summaries:
        lines.append(
            f"| `{run.get('run', '')}` | {run.get('backbone', '')} | {run.get('objective', '')} | "
            f"{int(float(run.get('best_step', 0) or 0))} | {float(run.get('best_val_loss', 0.0) or 0.0):.4f} | "
            f"{int(float(run.get('embedding_count', 0) or 0))} | {float(run.get('selection_score', 0.0) or 0.0):.4f} |"
        )
    if not run_summaries:
        lines.append("| no runs found | | | | | | |")
    lines.extend(["", "## Clustering Results", ""])
    lines.extend(["| Run | Method | Clusters | Noise | Silhouette | Davies-Bouldin |", "|---|---|---:|---:|---:|---:|"])
    for row in cluster_metrics[:40]:
        lines.append(
            f"| `{row.get('run', '')}` | {row.get('cluster_method', '')} | {int(float(row.get('n_clusters', 0) or 0))} | "
            f"{float(row.get('noise_fraction', 0.0) or 0.0):.3f} | {float(row.get('silhouette', 0.0) or 0.0):.3f} | "
            f"{float(row.get('davies_bouldin', 0.0) or 0.0):.3f} |"
        )
    if not cluster_metrics:
        lines.append("| no clustering evaluation found | | | | | |")
    lines.extend(["", "## Plots", ""])
    if plot_paths:
        for label, path in plot_paths.items():
            lines.append(f"![{label}]({Path(path).as_posix()})")
            lines.append("")
    else:
        lines.append("No plot assets were generated for this run.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation Checklist",
            "",
            "- Prefer HDBSCAN for first-pass family discovery because it can leave ambiguous particles as noise.",
            "- Compare discovered families by hit-count bucket, time span, total energy, PCA descriptors, and prototype galleries before assigning names.",
            "- Train supervised imitator classifiers only after a discovered clustering is frozen; imitator accuracy is a deployment metric, not discovery truth.",
            "- Treat file-scale or diffuse large clusters as quality-flagged candidates that may need separator review before physical interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


__all__ = [
    "Phase2DatasetConfig",
    "Phase2ModelConfig",
    "Phase2ParticleDataset",
    "Phase2TrainConfig",
    "build_phase2_dataset",
    "collate_phase2",
    "evaluate_phase2_experiment",
    "generate_phase2_report",
    "phase2_loss",
    "train_phase2_sweep",
]

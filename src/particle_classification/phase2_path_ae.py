from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .data.phase2_dataset import load_phase2_manifest


PATH_CHANNEL_NAMES = ["x_can", "y_can", "t_norm", "energy"]


@dataclass(frozen=True)
class PathAEConfig:
    latent_dim: int = 8
    hidden_dim: int = 192
    path_points: int = 128
    input_dim: int = 5
    output_dim: int = 4
    dropout: float = 0.03
    fourier_frequencies: int = 6


@dataclass(frozen=True)
class PathAETrainConfig:
    seed: int = 20260505
    steps: int = 10_000
    batch_sizes: tuple[int, ...] = (256, 512)
    learning_rate: float = 8e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    eval_interval: int = 250
    patience: int = 18
    min_steps: int = 3_000
    amp: bool = True
    lbfgs_steps: int = 15
    lbfgs_batch: int = 2048
    rotation_augment: bool = True
    jitter: float = 0.01
    dropout: float = 0.02


def weighted_theta_xy(x: np.ndarray, y: np.ndarray, energy: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.clip(np.asarray(energy, dtype=np.float64), 0.0, None) + 1e-3
    if x.size < 3:
        return 0.0, 0.0
    weights = weights / max(float(weights.sum()), 1e-12)
    cx = float(np.sum(x * weights))
    cy = float(np.sum(y * weights))
    centered = np.column_stack([x - cx, y - cy])
    cov = (centered * weights[:, None]).T @ centered
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    theta = float(math.atan2(vec[1], vec[0]))
    denom = float(vals.sum()) + 1e-12
    q_theta = float(max(0.0, min(1.0, (vals[0] - vals[-1]) / denom)))
    return theta, q_theta


def rotate_xy(x: np.ndarray, y: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x - s * y, s * x + c * y


def canonical_energy_path(points: np.ndarray, *, path_points: int = 128) -> tuple[np.ndarray, dict[str, float]]:
    """Convert Phase 2 raw point features into a canonical ordered energy path.

    The returned path has columns `(x_can, y_can, t_norm, energy)`.
    Detector-plane orientation is removed and stored as `theta_xy` metadata.
    """

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"Expected [N, >=5] point features, got {arr.shape}")
    x = arr[:, 0].astype(np.float64)
    y = arr[:, 1].astype(np.float64)
    t_norm = np.clip(arr[:, 3].astype(np.float64), 0.0, 1.0)
    energy = np.clip(arr[:, 4].astype(np.float64), 0.0, None)
    weights = energy + 1e-3
    weights = weights / max(float(weights.sum()), 1e-12)
    cx = float(np.sum(x * weights))
    cy = float(np.sum(y * weights))
    x0 = x - cx
    y0 = y - cy
    theta, q_theta = weighted_theta_xy(x0, y0, energy)
    x_can, y_can = rotate_xy(x0, y0, -theta)

    # Resolve the 180-degree PCA ambiguity using time direction, preserving time pitch in the canonical path.
    if x_can.size >= 2:
        corr = float(np.sum((x_can - np.average(x_can, weights=weights)) * (t_norm - np.average(t_norm, weights=weights)) * weights))
        if corr < 0:
            x_can = -x_can
            y_can = -y_can
            theta = theta + math.pi
    theta = float(math.atan2(math.sin(theta), math.cos(theta)))

    scale_xy = float(np.percentile(np.sqrt(x_can * x_can + y_can * y_can), 95)) if x_can.size else 1.0
    scale_xy = max(scale_xy, 1e-3)
    x_can = x_can / scale_xy
    y_can = y_can / scale_xy

    order = np.lexsort((x_can, y_can, t_norm))
    x_sorted = x_can[order]
    y_sorted = y_can[order]
    t_sorted = t_norm[order]
    e_sorted = energy[order]
    if x_sorted.size == 1:
        source_u = np.asarray([0.0], dtype=np.float64)
    else:
        source_u = np.linspace(0.0, 1.0, num=x_sorted.size, dtype=np.float64)
    target_u = np.linspace(0.0, 1.0, num=path_points, dtype=np.float64)
    path = np.column_stack(
        [
            np.interp(target_u, source_u, x_sorted),
            np.interp(target_u, source_u, y_sorted),
            np.interp(target_u, source_u, t_sorted),
            np.interp(target_u, source_u, e_sorted),
        ]
    ).astype(np.float32)
    path[:, 0:2] = np.clip(path[:, 0:2], -4.0, 4.0)
    path[:, 2:4] = np.clip(path[:, 2:4], 0.0, 1.5)
    metadata = {
        "theta_xy": theta,
        "q_theta": q_theta,
        "scale_xy": scale_xy,
        "centroid_x": cx,
        "centroid_y": cy,
        "n_hits": int(arr.shape[0]),
        "energy_sum": float(np.sum(energy)),
        "energy_max": float(np.max(energy)) if energy.size else 0.0,
    }
    return path, metadata


def build_path_cache(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    path_points: int = 128,
    max_items_by_split: dict[str, int | None] | None = None,
    seed: int = 20260505,
) -> dict[str, object]:
    dataset = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    max_items_by_split = max_items_by_split or {"train": None, "val": None, "test": None}
    summary: dict[str, object] = {"dataset": dataset.as_posix(), "output": output.as_posix(), "path_points": path_points, "splits": {}}
    for split in ["train", "val", "test"]:
        rows = load_phase2_manifest(dataset / "manifest.csv", split=split, max_items=max_items_by_split.get(split), sample_seed=seed)
        paths: list[np.ndarray | None] = [None] * len(rows)
        meta_rows: list[dict[str, object] | None] = [None] * len(rows)
        rows_by_chunk: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
        for idx, row in enumerate(rows):
            rows_by_chunk[str(row["chunk_path"])].append((idx, row))

        for chunk_idx, (chunk_path, chunk_rows) in enumerate(rows_by_chunk.items(), start=1):
            with np.load(chunk_path, allow_pickle=False) as data:
                chunk = {key: data[key] for key in data.files}
            for idx, row in chunk_rows:
                chunk_row = int(row["chunk_row"])
                start = int(chunk["offsets"][chunk_row])
                end = int(chunk["offsets"][chunk_row + 1])
                point_features = chunk["points"][start:end].astype(np.float32)
                path, metadata = canonical_energy_path(point_features, path_points=path_points)
                paths[idx] = path
                meta_rows[idx] = {
                    **metadata,
                    "row": idx,
                    "split": split,
                    "source_path": row.get("source_path", ""),
                    "particle_id": row.get("particle_id", ""),
                    "view_id": row.get("view_id", ""),
                    "source_n_hits": row.get("n_hits", metadata["n_hits"]),
                    "view_n_hits": row.get("view_n_hits", metadata["n_hits"]),
                    "path_n_samples": metadata["n_hits"],
                    "size_bucket": row.get("size_bucket", ""),
                }
            if chunk_idx % 25 == 0 or chunk_idx == len(rows_by_chunk):
                print(
                    f"[path-cache] split={split} chunks={chunk_idx}/{len(rows_by_chunk)} "
                    f"rows={sum(len(value) for value in list(rows_by_chunk.values())[:chunk_idx])}/{len(rows)}",
                    flush=True,
                )

        ready_paths = [path for path in paths if path is not None]
        ready_meta_rows = [row for row in meta_rows if row is not None]
        arr = np.stack(ready_paths, axis=0).astype(np.float32) if ready_paths else np.empty((0, path_points, len(PATH_CHANNEL_NAMES)), dtype=np.float32)
        np.savez_compressed(output / f"{split}.npz", path=arr, channel_names=np.asarray(PATH_CHANNEL_NAMES))
        write_csv(output / f"{split}_metadata.csv", ready_meta_rows)
        summary["splits"][split] = {"items": int(arr.shape[0]), "metadata": (output / f"{split}_metadata.csv").as_posix()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PathArrayDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        with np.load(self.cache_dir / f"{split}.npz", allow_pickle=False) as data:
            self.path = data["path"].astype(np.float32)

    def __len__(self) -> int:
        return int(self.path.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.path[idx])


class CanonicalPathAutoencoder(nn.Module):
    def __init__(self, config: PathAEConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.hit_mlp = nn.Sequential(
            nn.Linear(config.input_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(h, h, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(h, h, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.attention = nn.Linear(h, 1)
        self.to_latent = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.latent_dim),
        )
        query_dim = 1 + 2 * config.fourier_frequencies
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim + query_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, config.output_dim),
        )
        self.apply(init_weights)

    def encode(self, path: torch.Tensor) -> torch.Tensor:
        u = path_query(path.shape[0], path.shape[1], device=path.device, dtype=path.dtype)
        h = self.hit_mlp(torch.cat([path, u[..., :1]], dim=-1))
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        weights = torch.softmax(self.attention(h).squeeze(-1), dim=1).unsqueeze(-1)
        att_pool = (h * weights).sum(dim=1)
        return self.to_latent(torch.cat([mean_pool, max_pool, att_pool], dim=-1))

    def decode(self, z: torch.Tensor, *, n_points: int | None = None) -> torch.Tensor:
        n_points = int(n_points or self.config.path_points)
        query = path_query(z.shape[0], n_points, device=z.device, dtype=z.dtype, frequencies=self.config.fourier_frequencies)
        z_expand = z.unsqueeze(1).expand(-1, n_points, -1)
        raw = self.decoder(torch.cat([z_expand, query], dim=-1))
        xy = torch.tanh(raw[..., 0:2]) * 3.0
        time_energy = torch.sigmoid(raw[..., 2:4]) * 1.25
        return torch.cat([xy, time_energy], dim=-1)

    def forward(self, path: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(path)
        recon = self.decode(z, n_points=path.shape[1])
        return {"z": z, "reconstruction": recon}


def path_query(
    batch: int,
    n_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    frequencies: int = 0,
) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, steps=n_points, device=device, dtype=dtype).view(1, n_points, 1).expand(batch, -1, -1)
    if frequencies <= 0:
        return u
    feats = [u]
    for idx in range(frequencies):
        freq = float(2**idx) * math.pi
        feats.append(torch.sin(freq * u))
        feats.append(torch.cos(freq * u))
    return torch.cat(feats, dim=-1)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def augment_path(path: torch.Tensor, *, jitter: float, dropout: float) -> torch.Tensor:
    out = path.clone()
    if jitter > 0:
        noise = torch.randn_like(out) * jitter
        noise[..., 2] *= 0.3
        noise[..., 3] *= 0.5
        out = out + noise
    if dropout > 0 and path.shape[1] > 4:
        keep = torch.rand(path.shape[:2], device=path.device) > dropout
        keep[:, 0] = True
        keep[:, -1] = True
        nearest = out.clone()
        for idx in range(1, path.shape[1]):
            nearest[:, idx] = torch.where(keep[:, idx : idx + 1], out[:, idx], nearest[:, idx - 1])
        out = nearest
    out[..., 0:2] = torch.clamp(out[..., 0:2], -4.0, 4.0)
    out[..., 2:4] = torch.clamp(out[..., 2:4], 0.0, 1.5)
    return out


def energy_path_tensor(path: torch.Tensor) -> torch.Tensor:
    energy = torch.clamp(path[..., 3:4], min=0.0)
    return torch.cat([path[..., 0:1] * energy, path[..., 1:2] * energy, path[..., 2:3] * energy, energy], dim=-1)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    num = torch.sqrt(torch.mean((pred - target).pow(2), dim=(1, 2)) + eps)
    denom = torch.sqrt(torch.mean(target.pow(2), dim=(1, 2)) + eps)
    return num / denom


def path_ae_loss(recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    channel_weights = torch.tensor([1.0, 1.0, 0.7, 1.4], device=target.device, dtype=target.dtype)
    weighted_recon = recon * channel_weights
    weighted_target = target * channel_weights
    path_rel = relative_l2(weighted_recon, weighted_target).mean()
    energy_rel = relative_l2(energy_path_tensor(recon), energy_path_tensor(target)).mean()
    diff_rel = relative_l2(recon[:, 1:] - recon[:, :-1], target[:, 1:] - target[:, :-1]).mean()
    energy_sum_rel = torch.mean(torch.abs(recon[..., 3].sum(dim=1) - target[..., 3].sum(dim=1)) / torch.clamp(target[..., 3].sum(dim=1), min=1e-4))
    time_monotonic = torch.relu(-(recon[:, 1:, 2] - recon[:, :-1, 2])).mean()
    loss = path_rel + 0.9 * energy_rel + 0.25 * diff_rel + 0.1 * energy_sum_rel + 0.05 * time_monotonic
    return loss, {
        "loss": float(loss.detach().cpu()),
        "path_relative_l2": float(path_rel.detach().cpu()),
        "energy_path_relative_l2": float(energy_rel.detach().cpu()),
        "diff_relative_l2": float(diff_rel.detach().cpu()),
        "energy_sum_relative_l1": float(energy_sum_rel.detach().cpu()),
        "time_monotonic_penalty": float(time_monotonic.detach().cpu()),
    }


def batch_size_for_step(batch_sizes: tuple[int, ...], step: int, total_steps: int) -> int:
    if not batch_sizes:
        return 256
    phase = min(len(batch_sizes) - 1, int((step - 1) / max(total_steps, 1) * len(batch_sizes)))
    return int(batch_sizes[phase])


def learning_rate_for_step(config: PathAETrainConfig, step: int) -> float:
    warmup = max(50, int(config.steps * 0.05))
    if step <= warmup:
        return config.learning_rate * step / warmup
    progress = (step - warmup) / max(config.steps - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def random_batch(data: np.ndarray, batch_size: int, device: str) -> torch.Tensor:
    idx = np.random.randint(0, data.shape[0], size=batch_size)
    return torch.from_numpy(data[idx]).to(device=device, dtype=torch.float32)


@torch.no_grad()
def evaluate_path_ae(model: CanonicalPathAutoencoder, data: np.ndarray, *, batch_size: int, device: str) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for start in range(0, data.shape[0], batch_size):
        target = torch.from_numpy(data[start : start + batch_size]).to(device=device, dtype=torch.float32)
        recon = model(target)["reconstruction"]
        _, metrics = path_ae_loss(recon, target)
        rows.append(metrics)
    if not rows:
        return {"loss": 0.0}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train_path_ae_run(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    model_config: PathAEConfig,
    train_config: PathAETrainConfig,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, object]:
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cache = Path(cache_dir)
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = np.load(cache / "train.npz", allow_pickle=False)["path"].astype(np.float32)
    val = np.load(cache / "val.npz", allow_pickle=False)["path"].astype(np.float32)
    if val.size == 0:
        val = train[: min(train.shape[0], 4096)]
    model = CanonicalPathAutoencoder(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available()))
    best_state = None
    best_metric = float("inf")
    best_step = 0
    bad = 0
    rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for step in range(1, train_config.steps + 1):
        lr = learning_rate_for_step(train_config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_size = batch_size_for_step(train_config.batch_sizes, step, train_config.steps)
        target = random_batch(train, batch_size, device)
        model.train()
        model_input = augment_path(target, jitter=train_config.jitter, dropout=train_config.dropout) if train_config.rotation_augment else target
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(train_config.amp and str(device).startswith("cuda") and torch.cuda.is_available())):
            model_output = model(model_input)
            loss, metrics = path_ae_loss(model_output["reconstruction"], target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % train_config.eval_interval == 0 or step == train_config.steps:
            val_metrics = evaluate_path_ae(model, val[: min(val.shape[0], 8192)], batch_size=512, device=device)
            row = {
                "step": step,
                "batch_size": batch_size,
                "lr": lr,
                **metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            rows.append(row)
            write_csv(run_dir / "metrics.csv", rows)
            monitor = float(val_metrics["energy_path_relative_l2"])
            if verbose:
                print(
                    f"step {step}/{train_config.steps}: energy_rel={metrics['energy_path_relative_l2']:.4f}, "
                    f"val_energy_rel={monitor:.4f}, lr={lr:.2e}, batch={batch_size}",
                    flush=True,
                )
            if monitor < best_metric:
                best_metric = monitor
                best_step = step
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if step >= train_config.min_steps and bad >= train_config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    lbfgs_metrics = {}
    if train_config.lbfgs_steps > 0 and train.shape[0] > 0:
        model.train()
        lbfgs_batch = random_batch(train, min(train_config.lbfgs_batch, train.shape[0]), device)
        optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.15, max_iter=train_config.lbfgs_steps, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad(set_to_none=True)
            recon = model(lbfgs_batch)["reconstruction"]
            lbfgs_loss, _ = path_ae_loss(recon, lbfgs_batch)
            lbfgs_loss.backward()
            return lbfgs_loss

        optimizer_lbfgs.step(closure)
        val_metrics = evaluate_path_ae(model, val[: min(val.shape[0], 8192)], batch_size=512, device=device)
        lbfgs_metrics = {f"lbfgs_val_{key}": value for key, value in val_metrics.items()}
        if val_metrics["energy_path_relative_l2"] < best_metric:
            best_metric = float(val_metrics["energy_path_relative_l2"])
            best_step = int(rows[-1]["step"]) if rows else 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test = np.load(cache / "test.npz", allow_pickle=False)["path"].astype(np.float32)
    test_metrics = evaluate_path_ae(model, test[: min(test.shape[0], 16384)], batch_size=512, device=device) if test.size else {}
    checkpoint = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "best_step": best_step,
            "best_val_energy_path_relative_l2": best_metric,
            "test_metrics": test_metrics,
        },
        checkpoint,
    )
    summary = {
        "run": run_dir.name,
        "checkpoint": checkpoint.as_posix(),
        "metrics": (run_dir / "metrics.csv").as_posix(),
        "latent_dim": model_config.latent_dim,
        "path_points": model_config.path_points,
        "best_step": best_step,
        "best_val_energy_path_relative_l2": best_metric,
        "duration_s": time.perf_counter() - start_time,
        **lbfgs_metrics,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_path_ae_checkpoint(path: str | Path, device: str = "cpu") -> CanonicalPathAutoencoder:
    checkpoint = torch.load(path, map_location="cpu")
    model = CanonicalPathAutoencoder(PathAEConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


__all__ = [
    "PATH_CHANNEL_NAMES",
    "PathAEConfig",
    "PathAETrainConfig",
    "CanonicalPathAutoencoder",
    "canonical_energy_path",
    "build_path_cache",
    "energy_path_tensor",
    "path_ae_loss",
    "train_path_ae_run",
    "load_path_ae_checkpoint",
]

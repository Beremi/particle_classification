from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from ...data.dump import load_sparse_json_dump, sparse_record_to_hits
from ...dbscan.reference import candidates_from_dbscan, cluster_candidate_hits
from ...geometry import ParticleCandidate, canonicalize_xy, rotate_xy, weighted_pca_geometry
from .model import XYInvariantParticleNet, XYInvariantParticleNetConfig


class CandidateDataset(Dataset):
    def __init__(self, candidates: Iterable[ParticleCandidate], *, canonicalize: bool = True):
        self.candidates = list(candidates)
        self.canonicalize = canonicalize

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> ParticleCandidate:
        candidate = self.candidates[index]
        return canonicalize_xy(candidate) if self.canonicalize else candidate


def candidate_features(candidate: ParticleCandidate) -> np.ndarray:
    """Normalize one candidate into model features `x, y, relative_t, log1p(e)`."""

    hits = candidate.hits.copy()
    if hits.shape[0] == 0:
        return np.zeros((1, 4), dtype=np.float32)
    geom = weighted_pca_geometry(candidate)
    hits[:, 0] = (hits[:, 0] - geom.centroid_xy[0]) / 255.0
    hits[:, 1] = (hits[:, 1] - geom.centroid_xy[1]) / 255.0
    time_span = float(np.max(hits[:, 2]) - np.min(hits[:, 2]))
    hits[:, 2] = 0.0 if time_span <= 0 else (hits[:, 2] - np.min(hits[:, 2])) / time_span
    hits[:, 3] = np.log1p(np.clip(hits[:, 3], a_min=0.0, a_max=None))
    return hits.astype(np.float32)


def collate_candidates(candidates: list[ParticleCandidate]) -> dict[str, torch.Tensor]:
    arrays = [candidate_features(candidate) for candidate in candidates]
    max_len = max(array.shape[0] for array in arrays) if arrays else 1
    features = np.zeros((len(arrays), max_len, 4), dtype=np.float32)
    mask = np.zeros((len(arrays), max_len), dtype=bool)
    for i, array in enumerate(arrays):
        features[i, : array.shape[0], :] = array
        mask[i, : array.shape[0]] = True
    return {"features": torch.from_numpy(features), "mask": torch.from_numpy(mask)}


def rotate_candidates(candidates: list[ParticleCandidate], angles: np.ndarray | None = None) -> list[ParticleCandidate]:
    if angles is None:
        angles = np.random.uniform(-np.pi, np.pi, size=len(candidates))
    return [rotate_xy(candidate, float(angle)) for candidate, angle in zip(candidates, angles, strict=True)]


def xy_invariance_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    return torch.mean((z_a - z_b) ** 2)


def circular_pose_loss(pred_unit: torch.Tensor, target_angle: torch.Tensor, *, undirected: bool = False) -> torch.Tensor:
    factor = 2.0 if undirected else 1.0
    target = torch.stack([torch.cos(factor * target_angle), torch.sin(factor * target_angle)], dim=-1)
    pred = torch.nn.functional.normalize(pred_unit, dim=-1)
    return torch.mean((pred - target) ** 2)


def load_bootstrap_candidates(config: dict) -> list[ParticleCandidate]:
    data_config = config.get("data", {})
    records = load_sparse_json_dump(data_config.get("dump_path", "data/matrix_dump_0001.txt"))
    max_records = int(data_config.get("max_records", 16))
    eps = float(data_config.get("dbscan_eps", 1.5))
    min_samples = int(data_config.get("dbscan_min_samples", 3))
    out: list[ParticleCandidate] = []
    for record in records[:max_records]:
        parent = ParticleCandidate(
            hits=sparse_record_to_hits(record),
            candidate_id=f"sample-{record.sample}-set-{record.set_index}",
            sample=record.sample,
            set_index=record.set_index,
        )
        result = cluster_candidate_hits(parent, eps=eps, min_samples=min_samples)
        out.extend(candidates_from_dbscan(parent, result))
    return [candidate for candidate in out if candidate.n_hits > 0]


def train_baseline_from_config(config_path: str | Path) -> dict[str, float | str | int]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    candidates = load_bootstrap_candidates(config)
    if not candidates:
        raise RuntimeError("No bootstrap candidates were built from the configured data.")

    model_config = XYInvariantParticleNetConfig(**config.get("model", {}))
    model = XYInvariantParticleNet(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config.get("learning_rate", 1e-3)))
    batch_size = int(training_config.get("batch_size", 16))
    steps = int(training_config.get("steps", 5))
    last_loss = 0.0
    for _ in range(steps):
        batch_candidates = random.sample(candidates, k=min(batch_size, len(candidates)))
        angles = np.random.uniform(-np.pi, np.pi, size=len(batch_candidates))
        rotated = rotate_candidates(batch_candidates, angles)
        batch_a = collate_candidates([canonicalize_xy(candidate) for candidate in batch_candidates])
        batch_b = collate_candidates([canonicalize_xy(candidate) for candidate in rotated])
        out_a = model(batch_a["features"], batch_a["mask"])
        out_b = model(batch_b["features"], batch_b["mask"])
        loss = xy_invariance_loss(out_a["z_class"], out_b["z_class"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    output_dir = Path(training_config.get("output_dir", "local_data/experiments/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "xy_invariant_pointnet.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    return {
        "candidates": len(candidates),
        "steps": steps,
        "last_loss": last_loss,
        "checkpoint": str(checkpoint_path),
    }

#!/usr/bin/env python3
"""Fit a pose-separated PCA/basis autoencoder for Phase 2 paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Pose path cache from train_phase2_pose_autoencoder.py")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for PCA basis artifacts.")
    parser.add_argument("--latent-dim", type=int, action="append", default=None)
    parser.add_argument("--fit-items", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260505)
    return parser.parse_args()


def energy_path(path: np.ndarray) -> np.ndarray:
    energy = np.clip(path[..., 3:4], 0.0, None)
    return np.concatenate([path[..., 0:1] * energy, path[..., 1:2] * energy, path[..., 2:3] * energy, energy], axis=-1)


def relative_energy_path_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_ep = energy_path(pred)
    target_ep = energy_path(target)
    num = np.sqrt(np.mean((pred_ep - target_ep) ** 2, axis=(1, 2)) + 1e-6)
    denom = np.sqrt(np.mean(target_ep**2, axis=(1, 2)) + 1e-6)
    return num / denom


def reconstruct_from_components(flat: np.ndarray, mean: np.ndarray, components: np.ndarray, dim: int, shape: tuple[int, int]) -> np.ndarray:
    z = (flat - mean) @ components[:dim].T
    recon = (z @ components[:dim] + mean).reshape(flat.shape[0], *shape).astype(np.float32)
    recon[..., 0:2] = np.clip(recon[..., 0:2], -4.0, 4.0)
    recon[..., 2:4] = np.clip(recon[..., 2:4], 0.0, 1.5)
    return recon


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dims = sorted(set(args.latent_dim or [8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512]))
    max_dim = max(dims)
    with np.load(args.cache / "train.npz", allow_pickle=False) as data:
        train = data["canonical"].astype(np.float32)
    with np.load(args.cache / "val.npz", allow_pickle=False) as data:
        val = data["canonical"].astype(np.float32)
    with np.load(args.cache / "test.npz", allow_pickle=False) as data:
        test = data["canonical"].astype(np.float32)

    rng = np.random.default_rng(args.seed)
    fit_items = min(args.fit_items, train.shape[0])
    fit_idx = rng.choice(train.shape[0], size=fit_items, replace=False)
    fit_flat = train[fit_idx].reshape(fit_items, -1)
    pca = PCA(n_components=max_dim, svd_solver="full")
    pca.fit(fit_flat)
    mean = pca.mean_.astype(np.float32)
    components = pca.components_.astype(np.float32)
    np.savez_compressed(
        args.out / f"pose_pca_basis_z{max_dim}.npz",
        mean=mean,
        components=components,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        path_shape=np.asarray(train.shape[1:], dtype=np.int32),
    )

    rows: list[dict[str, object]] = []
    for dim in dims:
        row: dict[str, object] = {
            "latent_dim": dim,
            "fit_items": fit_items,
            "explained_variance_ratio": float(pca.explained_variance_ratio_[:dim].sum()),
        }
        for split, arr in [("val", val), ("test", test)]:
            flat = arr.reshape(arr.shape[0], -1)
            recon = reconstruct_from_components(flat, mean, components, dim, arr.shape[1:])
            rel = relative_energy_path_l2(recon, arr)
            row[f"{split}_mean"] = float(np.mean(rel))
            row[f"{split}_median"] = float(np.median(rel))
            row[f"{split}_p90"] = float(np.percentile(rel, 90))
            row[f"{split}_p99"] = float(np.percentile(rel, 99))
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    pd.DataFrame(rows).to_csv(args.out / "pose_pca_sweep.csv", index=False)
    (args.out / "pose_pca_sweep.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

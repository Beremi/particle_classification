#!/usr/bin/env python3
"""Find and render voxel-AE particle pairs with nearest shape latents."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from particle_classification.experiments.autoencoders.voxel import (  # noqa: E402
    VoxelGridConfig,
    VoxelMLPConfig,
    VoxelPatchMLPAutoencoder,
    VoxelSparseCache,
    corrupt_voxel_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--pair-mode", choices=["nearest", "rotation"], default="nearest")
    parser.add_argument("--xy-pairs", type=int, default=5)
    parser.add_argument("--tilt-pairs", type=int, default=5)
    parser.add_argument("--neighbor-k", type=int, default=128)
    parser.add_argument("--min-xy-diff-deg", type=float, default=30.0)
    parser.add_argument("--min-tilt-diff-deg", type=float, default=10.0)
    parser.add_argument("--max-hit-ratio", type=float, default=3.0)
    parser.add_argument("--max-energy-ratio", type=float, default=4.0)
    parser.add_argument("--morphology", choices=["any", "line"], default="any")
    parser.add_argument("--min-linearity-xy", type=float, default=0.72)
    parser.add_argument("--min-aspect-xy", type=float, default=2.2)
    parser.add_argument("--min-xy-span", type=float, default=10.0)
    parser.add_argument("--candidate-count", type=int, default=12000)
    parser.add_argument("--min-hits", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-mode", choices=["clean", "corrupted"], default="clean")
    parser.add_argument("--blur-kernel", type=int, default=3)
    parser.add_argument("--blur-mix", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.004)
    parser.add_argument("--voxel-dropout", type=float, default=0.008)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> tuple[VoxelPatchMLPAutoencoder, VoxelGridConfig]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = VoxelMLPConfig(**checkpoint["model_config"])
    grid_config = VoxelGridConfig(**checkpoint["grid_config"])
    model = VoxelPatchMLPAutoencoder(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, grid_config


def four_time_slices(volume: np.ndarray) -> np.ndarray:
    chunks = np.array_split(volume, 4, axis=0)
    return np.stack([chunk.sum(axis=0) for chunk in chunks], axis=0)


def row_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except ValueError:
        return default


def row_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except ValueError:
        return default


def angle_diff_mod_pi(a: float, b: float) -> float:
    """Smallest orientation difference for axes where theta and theta+pi match."""

    return abs((a - b + np.pi / 2.0) % np.pi - np.pi / 2.0)


def weighted_orientation(volume: np.ndarray, grid_config: VoxelGridConfig) -> tuple[float, float, float, float]:
    """Estimate XY orientation and 3D time tilt from weighted PCA.

    ``theta_xy`` is the major-axis angle in the detector plane modulo pi.
    ``theta_time_tilt`` is the absolute angle of the 3D major axis away from
    the XY plane, using voxel coordinates scaled by the grid bin sizes.
    ``linearity_xy`` and ``aspect_xy`` summarize detector-plane elongation.
    """

    flat = volume.reshape(-1)
    active = np.flatnonzero(flat > 0)
    if active.size < 2:
        return 0.0, 0.0, 0.0, 1.0
    values = flat[active].astype(np.float64)
    t = active // (grid_config.y_bins * grid_config.x_bins)
    rem = active % (grid_config.y_bins * grid_config.x_bins)
    y = rem // grid_config.x_bins
    x = rem % grid_config.x_bins
    xy = np.column_stack([x.astype(np.float64) * grid_config.xy_bin, y.astype(np.float64) * grid_config.xy_bin])
    w = values / max(float(values.sum()), 1e-12)
    xy_centered = xy - np.sum(xy * w[:, None], axis=0)
    cov_xy = (xy_centered * w[:, None]).T @ xy_centered
    evals_xy, evecs_xy = np.linalg.eigh(cov_xy)
    evals_xy = np.maximum(evals_xy, 0.0)
    order_xy = np.argsort(evals_xy)
    minor_eval = float(evals_xy[order_xy[0]])
    major_eval = float(evals_xy[order_xy[-1]])
    major_xy = evecs_xy[:, int(order_xy[-1])]
    theta_xy = float(np.arctan2(major_xy[1], major_xy[0]) % np.pi)
    aspect_xy = float(np.sqrt((major_eval + 1e-12) / (minor_eval + 1e-12)))
    linearity_xy = float(1.0 - minor_eval / max(major_eval, 1e-12))

    xyz = np.column_stack(
        [
            x.astype(np.float64) * grid_config.xy_bin,
            y.astype(np.float64) * grid_config.xy_bin,
            t.astype(np.float64) * grid_config.time_bin,
        ]
    )
    xyz_centered = xyz - np.sum(xyz * w[:, None], axis=0)
    cov_xyz = (xyz_centered * w[:, None]).T @ xyz_centered
    evals_xyz, evecs_xyz = np.linalg.eigh(cov_xyz)
    major_xyz = evecs_xyz[:, int(np.argmax(evals_xyz))]
    xy_norm = float(np.linalg.norm(major_xyz[:2]))
    theta_time_tilt = float(np.arctan2(abs(float(major_xyz[2])), max(xy_norm, 1e-12)))
    return theta_xy, theta_time_tilt, linearity_xy, aspect_xy


def encode_rows(
    model: VoxelPatchMLPAutoencoder,
    dataset: VoxelSparseCache,
    rows: list[dict[str, str]],
    *,
    batch_size: int,
    device: torch.device,
    input_mode: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape_batches: list[np.ndarray] = []
    aux_batches: list[np.ndarray] = []
    energy_batches: list[np.ndarray] = []
    theta_xy_values: list[float] = []
    theta_time_tilt_values: list[float] = []
    linearity_xy_values: list[float] = []
    aspect_xy_values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            clean = dataset.rows_to_dense(batch_rows, device=device)
            clean_np = clean.detach().cpu().numpy()
            for volume in clean_np:
                theta_xy, theta_time_tilt, linearity_xy, aspect_xy = weighted_orientation(volume, dataset.grid_config)
                theta_xy_values.append(theta_xy)
                theta_time_tilt_values.append(theta_time_tilt)
                linearity_xy_values.append(linearity_xy)
                aspect_xy_values.append(aspect_xy)
            model_input = clean
            if input_mode == "corrupted":
                torch.manual_seed(args.seed + start)
                model_input = corrupt_voxel_batch(
                    clean,
                    blur_kernel=args.blur_kernel,
                    blur_mix=args.blur_mix,
                    noise_std=args.noise_std,
                    voxel_dropout=args.voxel_dropout,
                )
            z_shape, z_aux = model.encode(model_input)
            shape_batches.append(z_shape.detach().cpu().numpy().astype(np.float32))
            aux_batches.append(z_aux.detach().cpu().numpy().astype(np.float32))
            energy_batches.append(clean.reshape(clean.shape[0], -1).sum(dim=1).detach().cpu().numpy().astype(np.float32))
    return (
        np.vstack(shape_batches),
        np.vstack(aux_batches),
        np.concatenate(energy_batches),
        np.asarray(theta_xy_values, dtype=np.float32),
        np.asarray(theta_time_tilt_values, dtype=np.float32),
        np.asarray(linearity_xy_values, dtype=np.float32),
        np.asarray(aspect_xy_values, dtype=np.float32),
    )


def select_pairs(z_shape: np.ndarray, z_aux: np.ndarray, *, count: int, neighbor_k: int = 16) -> list[dict[str, float | int]]:
    shape_std = np.maximum(z_shape.std(axis=0), 1e-6)
    z_norm = (z_shape - z_shape.mean(axis=0)) / shape_std
    tree = cKDTree(z_norm)
    k = min(neighbor_k + 1, len(z_norm))
    distances, indices = tree.query(z_norm, k=k)
    candidates: list[tuple[float, int, int]] = []
    for i in range(len(z_norm)):
        for dist, j in zip(np.atleast_1d(distances[i])[1:], np.atleast_1d(indices[i])[1:], strict=False):
            j = int(j)
            if i == j:
                continue
            a, b = sorted((i, j))
            candidates.append((float(dist), a, b))
    candidates = sorted(set(candidates), key=lambda item: item[0])
    used: set[int] = set()
    pairs: list[dict[str, float | int]] = []
    for norm_dist, a, b in candidates:
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        pairs.append(
            {
                "a": a,
                "b": b,
                "shape_distance_zscore": norm_dist,
                "shape_distance_raw": float(np.linalg.norm(z_shape[a] - z_shape[b])),
                "aux_distance_raw": float(np.linalg.norm(z_aux[a] - z_aux[b])),
            }
        )
        if len(pairs) >= count:
            break
    return pairs


def candidate_neighbor_pairs(z_shape: np.ndarray, *, neighbor_k: int) -> list[tuple[float, int, int]]:
    shape_std = np.maximum(z_shape.std(axis=0), 1e-6)
    z_norm = (z_shape - z_shape.mean(axis=0)) / shape_std
    tree = cKDTree(z_norm)
    k = min(neighbor_k + 1, len(z_norm))
    distances, indices = tree.query(z_norm, k=k)
    candidates: set[tuple[float, int, int]] = set()
    for i in range(len(z_norm)):
        for dist, j in zip(np.atleast_1d(distances[i])[1:], np.atleast_1d(indices[i])[1:], strict=False):
            j = int(j)
            if i == j:
                continue
            a, b = sorted((i, j))
            candidates.add((float(dist), a, b))
    return sorted(candidates, key=lambda item: item[0])


def select_rotation_pairs(
    z_shape: np.ndarray,
    z_aux: np.ndarray,
    theta_xy: np.ndarray,
    theta_time_tilt: np.ndarray,
    hit_counts: np.ndarray,
    energies: np.ndarray,
    *,
    xy_count: int,
    tilt_count: int,
    neighbor_k: int,
    min_xy_diff_deg: float,
    min_tilt_diff_deg: float,
    max_hit_ratio: float,
    max_energy_ratio: float,
) -> list[dict[str, float | int | str]]:
    candidates = candidate_neighbor_pairs(z_shape, neighbor_k=neighbor_k)

    def comparable(a: int, b: int) -> bool:
        hit_ratio = max(float(hit_counts[a]), float(hit_counts[b])) / max(min(float(hit_counts[a]), float(hit_counts[b])), 1.0)
        energy_ratio = max(float(energies[a]), float(energies[b])) / max(min(float(energies[a]), float(energies[b])), 1e-12)
        return hit_ratio <= max_hit_ratio and energy_ratio <= max_energy_ratio

    def build_pair(norm_dist: float, a: int, b: int, category: str) -> dict[str, float | int | str]:
        xy_diff = angle_diff_mod_pi(float(theta_xy[a]), float(theta_xy[b]))
        tilt_diff = abs(float(theta_time_tilt[a]) - float(theta_time_tilt[b]))
        return {
            "a": a,
            "b": b,
            "category": category,
            "shape_distance_zscore": norm_dist,
            "shape_distance_raw": float(np.linalg.norm(z_shape[a] - z_shape[b])),
            "aux_distance_raw": float(np.linalg.norm(z_aux[a] - z_aux[b])),
            "theta_xy_a_deg": float(np.degrees(theta_xy[a])),
            "theta_xy_b_deg": float(np.degrees(theta_xy[b])),
            "theta_xy_diff_deg": float(np.degrees(xy_diff)),
            "theta_time_tilt_a_deg": float(np.degrees(theta_time_tilt[a])),
            "theta_time_tilt_b_deg": float(np.degrees(theta_time_tilt[b])),
            "theta_time_tilt_diff_deg": float(np.degrees(tilt_diff)),
            "hit_ratio": max(float(hit_counts[a]), float(hit_counts[b])) / max(min(float(hit_counts[a]), float(hit_counts[b])), 1.0),
            "energy_ratio": max(float(energies[a]), float(energies[b])) / max(min(float(energies[a]), float(energies[b])), 1e-12),
        }

    def choose(category: str, angle_key: str, min_diff: float, count: int, used_global: set[int]) -> list[dict[str, float | int | str]]:
        scored: list[tuple[float, dict[str, float | int | str]]] = []
        for norm_dist, a, b in candidates:
            if not comparable(a, b):
                continue
            pair = build_pair(norm_dist, a, b, category)
            diff = float(pair[angle_key])
            if diff < min_diff:
                continue
            score = norm_dist / (diff + 1e-6)
            scored.append((score, pair))
        if len(scored) < count:
            for norm_dist, a, b in candidates:
                if not comparable(a, b):
                    continue
                pair = build_pair(norm_dist, a, b, category)
                diff = float(pair[angle_key])
                score = norm_dist / (diff + 1e-6)
                scored.append((score, pair))
        selected: list[dict[str, float | int | str]] = []
        used_local: set[int] = set()
        for _, pair in sorted(scored, key=lambda item: (item[0], float(item[1]["shape_distance_zscore"]))):
            a = int(pair["a"])
            b = int(pair["b"])
            if a in used_local or b in used_local or a in used_global or b in used_global:
                continue
            used_local.add(a)
            used_local.add(b)
            used_global.add(a)
            used_global.add(b)
            selected.append(pair)
            if len(selected) >= count:
                break
        return selected

    used: set[int] = set()
    xy_pairs = choose("xy_rotation", "theta_xy_diff_deg", min_xy_diff_deg, xy_count, used)
    tilt_pairs = choose("time_tilt_rotation", "theta_time_tilt_diff_deg", min_tilt_diff_deg, tilt_count, used)
    return xy_pairs + tilt_pairs


def render_pair(
    volume_a: np.ndarray,
    volume_b: np.ndarray,
    recon_a: np.ndarray,
    recon_b: np.ndarray,
    shape_only_a: np.ndarray,
    shape_only_b: np.ndarray,
    row_a: dict[str, str],
    row_b: dict[str, str],
    pair: dict[str, float | int],
    path: Path,
) -> None:
    row_items = [
        (f"A original h{row_a['n_hits']}", four_time_slices(volume_a)),
        ("A recon full", four_time_slices(recon_a)),
        ("A shape-only", four_time_slices(shape_only_a)),
        (f"B original h{row_b['n_hits']}", four_time_slices(volume_b)),
        ("B recon full", four_time_slices(recon_b)),
        ("B shape-only", four_time_slices(shape_only_b)),
    ]
    vmax = max(max(float(slices.max()) for _, slices in row_items), 1e-6)
    fig, axes = plt.subplots(len(row_items), 4, figsize=(12.2, 14.2), constrained_layout=True)
    images = []
    for col in range(4):
        axes[0, col].set_title(f"time block {col + 1}")
        for row_idx, (label, slices) in enumerate(row_items):
            images.append(axes[row_idx, col].imshow(slices[col], origin="lower", cmap="magma", vmin=0, vmax=vmax))
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
            if col == 0:
                axes[row_idx, col].set_ylabel(label)
    fig.colorbar(images[0], ax=axes[:, :], shrink=0.8, label="summed normalized log energy")
    fig.suptitle(
        f"shape dist z={pair['shape_distance_zscore']:.4g}, raw={pair['shape_distance_raw']:.4g}, "
        f"aux dist={pair['aux_distance_raw']:.4g} | "
        f"dXY={float(pair.get('theta_xy_diff_deg', 0.0)):.1f} deg, "
        f"dTilt={float(pair.get('theta_time_tilt_diff_deg', 0.0)):.1f} deg",
        fontsize=11,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt_vec(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):+.3f}" for value in values) + "]"


def particle_summary(
    row: dict[str, str],
    energy: float,
    theta_xy: float,
    theta_time_tilt: float,
    linearity_xy: float,
    aspect_xy: float,
) -> str:
    return (
        f"hits `{row['n_hits']}`, E `{energy:.3f}`, "
        f"theta_xy `{np.degrees(theta_xy):.1f} deg`, theta_time_tilt `{np.degrees(theta_time_tilt):.1f} deg`, "
        f"linearity_xy `{linearity_xy:.3f}`, aspect_xy `{aspect_xy:.2f}`, "
        f"span x/y/t `{row_float(row, 'x_span'):.1f}`/`{row_float(row, 'y_span'):.1f}`/`{row_float(row, 'time_span'):.3f}`, "
        f"source `{row['source_path']}`, particle `{row['particle_id']}`"
    )


def main() -> None:
    args = parse_args()
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    model, grid_config = load_model(args.checkpoint, args.device)
    device = next(model.parameters()).device
    dataset = VoxelSparseCache(args.cache, split=args.split, grid_config=grid_config, seed=args.seed)
    candidates = [row for row in dataset.rows if row_int(row, "n_hits") >= args.min_hits]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: args.candidate_count]
    if len(candidates) < 2:
        raise SystemExit("Need at least two candidates.")
    z_shape, z_aux, energies, theta_xy, theta_time_tilt, linearity_xy, aspect_xy = encode_rows(
        model,
        dataset,
        candidates,
        batch_size=args.batch_size,
        device=device,
        input_mode=args.input_mode,
        args=args,
    )
    if args.morphology == "line":
        keep = np.asarray(
            [
                bool(
                    linearity_xy[idx] >= args.min_linearity_xy
                    and aspect_xy[idx] >= args.min_aspect_xy
                    and max(row_float(row, "x_span"), row_float(row, "y_span")) >= args.min_xy_span
                )
                for idx, row in enumerate(candidates)
            ],
            dtype=bool,
        )
        if int(np.count_nonzero(keep)) < 2:
            raise SystemExit(
                "Line-like filter found fewer than two candidates. "
                "Try reducing --min-linearity-xy, --min-aspect-xy, or --min-xy-span."
            )
        candidates = [row for row, flag in zip(candidates, keep, strict=True) if flag]
        z_shape = z_shape[keep]
        z_aux = z_aux[keep]
        energies = energies[keep]
        theta_xy = theta_xy[keep]
        theta_time_tilt = theta_time_tilt[keep]
        linearity_xy = linearity_xy[keep]
        aspect_xy = aspect_xy[keep]
    hit_counts = np.asarray([row_int(row, "n_hits") for row in candidates], dtype=np.float32)
    if args.pair_mode == "rotation":
        pairs = select_rotation_pairs(
            z_shape,
            z_aux,
            theta_xy,
            theta_time_tilt,
            hit_counts,
            energies,
            xy_count=args.xy_pairs,
            tilt_count=args.tilt_pairs,
            neighbor_k=args.neighbor_k,
            min_xy_diff_deg=args.min_xy_diff_deg,
            min_tilt_diff_deg=args.min_tilt_diff_deg,
            max_hit_ratio=args.max_hit_ratio,
            max_energy_ratio=args.max_energy_ratio,
        )
    else:
        pairs = select_pairs(z_shape, z_aux, count=args.pairs, neighbor_k=args.neighbor_k)
    selected_indices = sorted({int(pair["a"]) for pair in pairs} | {int(pair["b"]) for pair in pairs})
    selected_dense = dataset.rows_to_dense([candidates[index] for index in selected_indices], device=device)
    selected_z_shape = torch.as_tensor(z_shape[selected_indices], dtype=torch.float32, device=device)
    selected_z_aux = torch.as_tensor(z_aux[selected_indices], dtype=torch.float32, device=device)
    with torch.no_grad():
        selected_recon = model.decode(selected_z_shape, selected_z_aux)[:, 0].detach().cpu().numpy()
        selected_shape_only = model.decode(selected_z_shape, torch.zeros_like(selected_z_aux))[:, 0].detach().cpu().numpy()
    dense_by_index = {index: selected_dense[pos].detach().cpu().numpy() for pos, index in enumerate(selected_indices)}
    recon_by_index = {index: selected_recon[pos] for pos, index in enumerate(selected_indices)}
    shape_only_by_index = {index: selected_shape_only[pos] for pos, index in enumerate(selected_indices)}

    lines = ["# Phase 2 Voxel Shape-Latent Pair Gallery", ""]
    lines.append(f"Checkpoint: `{args.checkpoint.as_posix()}`")
    lines.append(f"Split: `{args.split}`")
    lines.append(f"Candidate search: `{len(candidates)}` particles with at least `{args.min_hits}` hits.")
    lines.append(f"Encoder input mode: `{args.input_mode}`")
    lines.append(f"Pair mode: `{args.pair_mode}`")
    lines.append(f"Morphology filter: `{args.morphology}`")
    if args.morphology == "line":
        lines.append(
            f"Line-like thresholds: `linearity_xy >= {args.min_linearity_xy}`, "
            f"`aspect_xy >= {args.min_aspect_xy}`, `max(x_span,y_span) >= {args.min_xy_span}`."
        )
    if args.pair_mode == "rotation":
        lines.append(
            f"Pair comparability: `hit_ratio <= {args.max_hit_ratio}`, `energy_ratio <= {args.max_energy_ratio}`."
        )
    if args.input_mode == "corrupted":
        lines.append(
            f"Corruption: `blur_kernel={args.blur_kernel}`, `blur_mix={args.blur_mix}`, "
            f"`noise_std={args.noise_std}`, `voxel_dropout={args.voxel_dropout}`."
        )
    lines.append("")
    lines.append(
        "Pairs are nearest neighbors in the z-scored 8D `z_shape` space. The 7 `z_aux` values are shown raw. "
        "In this checkpoint they are unconstrained auxiliary latents, not guaranteed physical dx/dy/theta/scale parameters."
    )
    lines.append("")
    lines.append(
        "`theta_xy` and `theta_time_tilt` are not network outputs here. They are diagnostic PCA angles measured "
        "from the centered voxel particle: `theta_xy` is the detector-plane major-axis angle modulo 180 degrees, "
        "and `theta_time_tilt` is the 3D major-axis angle away from the XY plane."
    )
    lines.append("")
    lines.append(
        "Each image contains original rows plus reconstructions. `recon full` decodes `z_shape + z_aux`. "
        "`shape-only` decodes the same `z_shape` with `z_aux=0`; this is a diagnostic proxy for "
        "\"before transform/aux influence\". It is not a true pre-transform tensor, because this checkpoint "
        "does not have a separate final transform layer."
    )
    lines.append("")
    lines.append(
        "| pair | category | shape dist z-scored | dXY deg | dTilt deg | "
        "linearity A/B | aspect A/B | hits A/B | E A/B | image |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    pair_records = []
    for pair_idx, pair in enumerate(pairs, start=1):
        a = int(pair["a"])
        b = int(pair["b"])
        image_path = args.asset_dir / f"shape_latent_pair_{pair_idx:02d}.png"
        render_pair(
            dense_by_index[a],
            dense_by_index[b],
            recon_by_index[a],
            recon_by_index[b],
            shape_only_by_index[a],
            shape_only_by_index[b],
            candidates[a],
            candidates[b],
            pair,
            image_path,
        )
        rel_image = image_path.relative_to(args.out.parent).as_posix()
        lines.append(
            f"| {pair_idx} | `{pair.get('category', 'nearest')}` | {pair['shape_distance_zscore']:.5f} | "
            f"{float(pair.get('theta_xy_diff_deg', 0.0)):.1f} | "
            f"{float(pair.get('theta_time_tilt_diff_deg', 0.0)):.1f} | "
            f"{linearity_xy[a]:.2f}/{linearity_xy[b]:.2f} | {aspect_xy[a]:.1f}/{aspect_xy[b]:.1f} | "
            f"{candidates[a]['n_hits']}/{candidates[b]['n_hits']} | "
            f"{energies[a]:.2f}/{energies[b]:.2f} | [png]({rel_image}) |"
        )
        pair_records.append((pair_idx, pair, a, b, rel_image))
    lines.append("")
    for pair_idx, pair, a, b, rel_image in pair_records:
        lines.append(f"## Pair {pair_idx}")
        lines.append("")
        lines.append(f"![pair {pair_idx}]({rel_image})")
        lines.append("")
        lines.append(f"- category: `{pair.get('category', 'nearest')}`")
        lines.append(
            f"- angle differences: theta_xy `{float(pair.get('theta_xy_diff_deg', 0.0)):.1f} deg`, "
            f"theta_time_tilt `{float(pair.get('theta_time_tilt_diff_deg', 0.0)):.1f} deg`"
        )
        lines.append(
            f"- comparability: hit_ratio `{float(pair.get('hit_ratio', 1.0)):.2f}`, "
            f"energy_ratio `{float(pair.get('energy_ratio', 1.0)):.2f}`"
        )
        lines.append(
            f"- A: {particle_summary(candidates[a], float(energies[a]), float(theta_xy[a]), float(theta_time_tilt[a]), float(linearity_xy[a]), float(aspect_xy[a]))}"
        )
        lines.append(
            f"- B: {particle_summary(candidates[b], float(energies[b]), float(theta_xy[b]), float(theta_time_tilt[b]), float(linearity_xy[b]), float(aspect_xy[b]))}"
        )
        lines.append(f"- `z_shape_A`: `{fmt_vec(z_shape[a])}`")
        lines.append(f"- `z_shape_B`: `{fmt_vec(z_shape[b])}`")
        lines.append(f"- `z_aux_A`: `{fmt_vec(z_aux[a])}`")
        lines.append(f"- `z_aux_B`: `{fmt_vec(z_aux[b])}`")
        lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

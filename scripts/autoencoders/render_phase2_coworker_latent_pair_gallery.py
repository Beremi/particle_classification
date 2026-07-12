#!/usr/bin/env python3
"""Render nearest-neighbor particle pairs in the current simple z8 latent space."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from particle_classification.experiments.autoencoders.canonical_voxel import _render_rows
from particle_classification.experiments.autoencoders.voxel import (
    VoxelGridConfig,
    VoxelMLPConfig,
    VoxelPatchMLPAutoencoder,
    VoxelSparseCache,
)


def load_model(path: Path, device: str) -> tuple[VoxelPatchMLPAutoencoder, VoxelGridConfig]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = VoxelMLPConfig(**dict(payload["model_config"]))
    grid_config = VoxelGridConfig(**dict(payload["grid_config"]))
    model = VoxelPatchMLPAutoencoder(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, grid_config


def load_rows(cache: Path, grid: VoxelGridConfig, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        dataset = VoxelSparseCache(cache, split=split, grid_config=grid, seed=seed)
        for row in dataset.rows:
            item = dict(row)
            item["split"] = split
            rows.append(item)
    return rows


def load_group_order(summary_csv: Path, limit: int) -> list[int]:
    out: list[int] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out.append(int(row["cluster"]))
            if len(out) >= limit:
                break
    return out


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).reshape(-1)) / max(np.linalg.norm(a.reshape(-1)), 1e-8))


def source_folder(path: str) -> str:
    parts = Path(path).parts
    return parts[0] if parts else path


def choose_pair(
    z_norm: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    cluster: int,
    *,
    min_hits: int,
    max_candidates: int,
    rng: np.random.Generator,
) -> tuple[int, int, float] | None:
    candidates = np.asarray(
        [
            idx
            for idx, row in enumerate(rows)
            if int(labels[idx]) == cluster and int(float(row.get("n_hits", 0))) >= min_hits
        ],
        dtype=np.int64,
    )
    if candidates.size < 2:
        return None
    if candidates.size > max_candidates:
        candidates = rng.choice(candidates, size=max_candidates, replace=False)

    z = z_norm[candidates]
    nn = NearestNeighbors(n_neighbors=min(20, len(candidates)), metric="euclidean")
    nn.fit(z)
    dist, ind = nn.kneighbors(z)
    best: tuple[int, int, float] | None = None
    for local_i in range(len(candidates)):
        global_i = int(candidates[local_i])
        for neighbor_pos in range(1, ind.shape[1]):
            local_j = int(ind[local_i, neighbor_pos])
            global_j = int(candidates[local_j])
            if rows[global_i]["source_path"] == rows[global_j]["source_path"]:
                continue
            hit_i = int(float(rows[global_i]["n_hits"]))
            hit_j = int(float(rows[global_j]["n_hits"]))
            if max(hit_i, hit_j) / max(1, min(hit_i, hit_j)) > 4.0:
                continue
            score = float(dist[local_i, neighbor_pos])
            if best is None or score < best[2]:
                best = (global_i, global_j, score)
                break
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("local_data/experiments/phase2_simple_voxel_ae_z8_b1024_clean_fixedmine_20phase_v001/checkpoint.pt"))
    parser.add_argument("--cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_8x32x32_representative_v001"))
    parser.add_argument("--latent-npz", type=Path, default=Path("local_data/experiments/phase2_simple_z8_latent_groups_v001/latents_and_groups_k32.npz"))
    parser.add_argument("--group-summary", type=Path, default=Path("local_data/experiments/phase2_simple_z8_latent_groups_v001/group_summary_k32.csv"))
    parser.add_argument("--out", type=Path, default=Path("experimental_notes/autoencoders/latent_analysis/phase2-coworker-similar-latent-pairs.md"))
    parser.add_argument("--asset-dir", type=Path, default=Path("experimental_notes/assets/phase2_coworker_similar_latent_pairs_v001"))
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--min-hits", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.latent_npz)
    z_norm = np.asarray(data["z_norm"], dtype=np.float32)
    z_shape = np.asarray(data["z_shape"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int32)

    model, grid = load_model(args.checkpoint, device)
    rows = load_rows(args.cache, grid, args.seed)
    if len(rows) != len(labels):
        raise ValueError(f"row/label mismatch: rows={len(rows)} labels={len(labels)}")

    datasets: dict[str, VoxelSparseCache] = {}
    def dense_for(row: dict[str, str]) -> torch.Tensor:
        split = row["split"]
        if split not in datasets:
            datasets[split] = VoxelSparseCache(args.cache, split=split, grid_config=grid, seed=args.seed)
        return datasets[split].rows_to_dense([row], device=device)

    rng = np.random.default_rng(args.seed)
    selected_clusters = load_group_order(args.group_summary, args.clusters)
    rendered: list[dict[str, object]] = []
    for pair_rank, cluster in enumerate(selected_clusters, start=1):
        pair = choose_pair(
            z_norm,
            labels,
            rows,
            cluster,
            min_hits=args.min_hits,
            max_candidates=args.max_candidates,
            rng=rng,
        )
        if pair is None:
            continue
        idx_a, idx_b, z_dist = pair
        row_a = rows[idx_a]
        row_b = rows[idx_b]
        dense_a = dense_for(row_a)
        dense_b = dense_for(row_b)
        batch = torch.cat([dense_a, dense_b], dim=0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
            recon = model(batch)["reconstruction"]
        target = batch.float().detach().cpu().numpy()
        recon_np = recon.float().detach().cpu().numpy()
        image = args.asset_dir / f"similar_latent_pair_{pair_rank:02d}_cluster_{cluster:02d}.png"
        _render_rows(
            [
                ("A original", target[0]),
                ("A reconstruction", recon_np[0]),
                ("B original", target[1]),
                ("B reconstruction", recon_np[1]),
                ("|A-B| original", np.abs(target[0] - target[1])),
            ],
            image,
            f"K32 cluster {cluster}: z distance {z_dist:.4f}",
        )
        rendered.append(
            {
                "rank": pair_rank,
                "cluster": cluster,
                "idx_a": idx_a,
                "idx_b": idx_b,
                "z_dist": z_dist,
                "hit_a": int(float(row_a.get("n_hits", 0))),
                "hit_b": int(float(row_b.get("n_hits", 0))),
                "source_a": row_a["source_path"],
                "source_b": row_b["source_path"],
                "folder_a": source_folder(row_a["source_path"]),
                "folder_b": source_folder(row_b["source_path"]),
                "particle_a": row_a["particle_id"],
                "particle_b": row_b["particle_id"],
                "recon_l2_a": relative_l2(target[0], recon_np[0]),
                "recon_l2_b": relative_l2(target[1], recon_np[1]),
                "raw_l2_ab": relative_l2(target[0], target[1]),
                "image": image.relative_to(args.out.parent).as_posix(),
                "z_shape_a": " ".join(f"{v:+.3f}" for v in z_shape[idx_a]),
                "z_shape_b": " ".join(f"{v:+.3f}" for v in z_shape[idx_b]),
            }
        )

    csv_path = args.asset_dir / "similar_latent_pairs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rendered[0].keys()) if rendered else ["rank"])
        writer.writeheader()
        writer.writerows(rendered)

    lines = [
        "# Phase 2 Similar-Latent Particle Pair Gallery",
        "",
        "Pairs are selected from the current clean simple z8 autoencoder. For each of the largest K=32 latent groups, "
        "the script finds a nearest-neighbor pair in standardized `z_shape` space, requiring different source files and at least 10 hits.",
        "",
        "| pair | K32 group | z distance | hits A/B | source folders | recon L2 A/B | original A-B rel L2 | image |",
        "|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rendered:
        lines.append(
            f"| {row['rank']} | {row['cluster']} | {float(row['z_dist']):.4f} | "
            f"{row['hit_a']}/{row['hit_b']} | `{row['folder_a']}` / `{row['folder_b']}` | "
            f"{float(row['recon_l2_a']):.3f}/{float(row['recon_l2_b']):.3f} | {float(row['raw_l2_ab']):.3f} | "
            f"[png]({row['image']}) |"
        )
    for row in rendered:
        lines.extend(
            [
                "",
                f"## Pair {row['rank']}: K32 Group {row['cluster']}",
                "",
                f"![pair {row['rank']}]({row['image']})",
                "",
                f"- A: `{row['source_a']}`, particle `{row['particle_a']}`, hits `{row['hit_a']}`",
                f"- B: `{row['source_b']}`, particle `{row['particle_b']}`, hits `{row['hit_b']}`",
                f"- z distance: `{float(row['z_dist']):.4f}`",
                f"- z_shape A: `{row['z_shape_a']}`",
                f"- z_shape B: `{row['z_shape_b']}`",
            ]
        )
    lines += [
        "",
        "## Notes",
        "",
        "- This gallery is diagnostic: similar latent codes do not guarantee physical identity.",
        "- The final panel is absolute original-tensor difference, so it exposes where two similar codes still differ in voxel space.",
        f"- CSV: `{csv_path.as_posix()}`",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from particle_classification.dbscan.reference import dbscan_labels
from particle_classification.data.t3pa import iter_t3pa_hits
from particle_classification.geometry import ParticleCandidate, weighted_pca_geometry


DEFAULT_INPUT = Path("local_data/raw/D05/tot_toa__r0000000030.t3pa")


@dataclass(frozen=True)
class ClusterStats:
    cluster_id: int
    n_hits: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float
    theta_xy: float
    q_theta: float
    phi_t: float
    principal_vector: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether 3D DBSCAN clusters look stable and whether paths may be split."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--eps", type=float, default=3.5)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--time-scale-m", type=float, default=20.0)
    parser.add_argument(
        "--near-factor",
        type=float,
        default=1.25,
        help="Report separate clusters whose closest scaled distance is within eps * near_factor.",
    )
    parser.add_argument("--max-pairs", type=int, default=25)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/D05_tot_toa_r0000000030_3d_dbscan_near_pairs.csv"),
    )
    args = parser.parse_args()

    data = load_hits(args.input)
    x, y, t, tot = data
    features = np.column_stack([x, y, t / args.time_scale_m])
    labels = dbscan_labels(features, eps=args.eps, min_samples=args.min_samples)

    cluster_ids = sorted(int(label) for label in set(labels.tolist()) if label != -1)
    noise = int(np.sum(labels == -1))
    stats = build_cluster_stats(cluster_ids, labels, x, y, t, tot, features)
    near_pairs = find_near_pairs(stats, labels, features, args.eps * args.near_factor)
    stability = parameter_stability(features, labels, args.eps, args.min_samples, args.time_scale_m, x, y, t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_near_pairs(args.out, near_pairs)

    sizes = np.array([item.n_hits for item in stats], dtype=int)
    path_like = [item for item in stats if item.n_hits >= 6 and item.q_theta >= 0.70]
    suspicious_splits = [
        item
        for item in near_pairs
        if item["orientation_similarity"] >= 0.85
        and item["cluster_a_q_theta"] >= 0.70
        and item["cluster_b_q_theta"] >= 0.70
    ]

    print(f"Input: {args.input}")
    print(
        f"DBSCAN: eps={args.eps}, min_samples={args.min_samples}, "
        f"time_scale={args.time_scale_m:g}M ToA ticks"
    )
    print(f"Hits: {len(labels):,}")
    print(f"Clusters: {len(cluster_ids):,}")
    print(f"Noise: {noise:,} ({noise / len(labels):.2%})")
    print(
        "Cluster size min/median/mean/max: "
        f"{sizes.min()}/{np.median(sizes):.1f}/{sizes.mean():.1f}/{sizes.max()}"
    )
    print(f"Path-like clusters (n_hits >= 6 and q_theta >= 0.70): {len(path_like):,}")
    print("")
    print("Parameter stability around this setting:")
    for row in stability:
        print(
            f"  eps={row['eps']:.2f}, time_scale={row['time_scale_m']:.1f}M: "
            f"clusters={row['clusters']:3d}, noise={row['noise']:4d}, ARI={row['ari']:.3f}"
        )
    print("")
    print(
        f"Near separate cluster pairs within eps * {args.near_factor:g}: "
        f"{len(near_pairs):,} total; {len(suspicious_splits):,} look path-like/aligned"
    )
    print(f"Near-pair CSV: {args.out}")
    print("")
    print("Top possible split candidates:")
    for row in suspicious_splits[: args.max_pairs]:
        print(
            f"  {row['cluster_a']:3d} <-> {row['cluster_b']:3d}: "
            f"closest={row['closest_scaled_distance']:.2f}, "
            f"sizes={row['cluster_a_hits']}/{row['cluster_b_hits']}, "
            f"orient_sim={row['orientation_similarity']:.2f}, "
            f"q={row['cluster_a_q_theta']:.2f}/{row['cluster_b_q_theta']:.2f}, "
            f"t_gap_m={row['time_gap_m']:.2f}"
        )


def load_hits(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hits = list(iter_t3pa_hits(path))
    if not hits:
        raise SystemExit(f"No hits found in {path}")
    x = np.array([hit.x for hit in hits], dtype=float)
    y = np.array([hit.y for hit in hits], dtype=float)
    toa = np.array([hit.toa for hit in hits], dtype=float)
    t = (toa - toa.min()) / 1_000_000.0
    tot = np.array([hit.tot for hit in hits], dtype=float)
    return x, y, t, tot


def build_cluster_stats(
    cluster_ids: list[int],
    labels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    tot: np.ndarray,
    features: np.ndarray,
) -> list[ClusterStats]:
    out = []
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        candidate = ParticleCandidate(
            np.column_stack([x[mask], y[mask], t[mask], np.log1p(tot[mask])]),
            candidate_id=str(cluster_id),
        )
        geom = weighted_pca_geometry(candidate)
        out.append(
            ClusterStats(
                cluster_id=cluster_id,
                n_hits=int(mask.sum()),
                x_min=float(x[mask].min()),
                x_max=float(x[mask].max()),
                y_min=float(y[mask].min()),
                y_max=float(y[mask].max()),
                t_min=float(t[mask].min()),
                t_max=float(t[mask].max()),
                theta_xy=geom.theta_xy,
                q_theta=geom.q_theta,
                phi_t=geom.phi_t,
                principal_vector=principal_vector(features[mask]),
            )
        )
    return out


def find_near_pairs(
    stats: list[ClusterStats],
    labels: np.ndarray,
    features: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int]]:
    by_id = {item.cluster_id: features[labels == item.cluster_id] for item in stats}
    rows = []
    for i, a in enumerate(stats):
        for b in stats[i + 1 :]:
            lower_bound = bbox_distance(a, b)
            if lower_bound > threshold:
                continue
            exact = closest_distance(by_id[a.cluster_id], by_id[b.cluster_id])
            if exact > threshold:
                continue
            orient_sim = float(abs(np.dot(a.principal_vector, b.principal_vector)))
            rows.append(
                {
                    "cluster_a": a.cluster_id,
                    "cluster_b": b.cluster_id,
                    "closest_scaled_distance": exact,
                    "cluster_a_hits": a.n_hits,
                    "cluster_b_hits": b.n_hits,
                    "orientation_similarity": orient_sim,
                    "cluster_a_q_theta": a.q_theta,
                    "cluster_b_q_theta": b.q_theta,
                    "time_gap_m": interval_gap(a.t_min, a.t_max, b.t_min, b.t_max),
                    "x_gap": interval_gap(a.x_min, a.x_max, b.x_min, b.x_max),
                    "y_gap": interval_gap(a.y_min, a.y_max, b.y_min, b.y_max),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["orientation_similarity"]),
            float(row["closest_scaled_distance"]),
            -min(int(row["cluster_a_hits"]), int(row["cluster_b_hits"])),
        ),
    )


def parameter_stability(
    base_features: np.ndarray,
    base_labels: np.ndarray,
    eps: float,
    min_samples: int,
    time_scale_m: float,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
) -> list[dict[str, float | int]]:
    rows = []
    for eps_factor, scale_factor in [
        (0.85, 1.0),
        (0.95, 1.0),
        (1.05, 1.0),
        (1.15, 1.0),
        (1.0, 0.75),
        (1.0, 1.50),
    ]:
        next_eps = eps * eps_factor
        next_time_scale = time_scale_m * scale_factor
        features = np.column_stack([x, y, t / next_time_scale])
        labels = dbscan_labels(features, eps=next_eps, min_samples=min_samples)
        rows.append(
            {
                "eps": next_eps,
                "time_scale_m": next_time_scale,
                "clusters": len(set(int(label) for label in labels.tolist()) - {-1}),
                "noise": int(np.sum(labels == -1)),
                "ari": adjusted_rand_index(base_labels, labels),
            }
        )
    return rows


def principal_vector(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0])
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    vec = vh[0]
    return vec / (np.linalg.norm(vec) + 1e-12)


def bbox_distance(a: ClusterStats, b: ClusterStats) -> float:
    return float(
        np.sqrt(
            interval_gap(a.x_min, a.x_max, b.x_min, b.x_max) ** 2
            + interval_gap(a.y_min, a.y_max, b.y_min, b.y_max) ** 2
            + interval_gap(a.t_min, a.t_max, b.t_min, b.t_max) ** 2
        )
    )


def closest_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))


def interval_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    if a_max < b_min:
        return float(b_min - a_max)
    if b_max < a_min:
        return float(a_min - b_max)
    return 0.0


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if labels_a.shape != labels_b.shape:
        raise ValueError("Label arrays must have the same shape.")
    n = labels_a.size
    if n < 2:
        return 1.0
    contingency = Counter(zip(labels_a.tolist(), labels_b.tolist(), strict=True))
    counts_a = Counter(labels_a.tolist())
    counts_b = Counter(labels_b.tolist())
    sum_comb = sum(comb2(count) for count in contingency.values())
    sum_a = sum(comb2(count) for count in counts_a.values())
    sum_b = sum(comb2(count) for count in counts_b.values())
    total = comb2(n)
    expected = sum_a * sum_b / total if total else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0
    return float((sum_comb - expected) / denominator)


def comb2(value: int) -> float:
    return value * (value - 1) / 2.0


def write_near_pairs(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

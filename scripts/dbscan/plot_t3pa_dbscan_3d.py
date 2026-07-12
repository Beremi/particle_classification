#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

from particle_classification.dbscan.reference import dbscan_labels
from particle_classification.data.t3pa import iter_t3pa_hits


DEFAULT_INPUT = Path("local_data/raw/D05/tot_toa__r0000000030.t3pa")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an interactive Matplotlib 3D DBSCAN view of a `.t3pa` export."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to a `.t3pa` file. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument("--eps", type=float, default=3.5, help="DBSCAN epsilon in scaled x/y/time space.")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples.")
    parser.add_argument(
        "--time-scale-m",
        type=float,
        default=20.0,
        help="How many million ToA ticks correspond to one spatial-pixel unit for DBSCAN.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional Matplotlib GUI backend, for example QtAgg or gtk4agg.",
    )
    args = parser.parse_args()

    plt = _import_pyplot(args.backend)

    hits = list(iter_t3pa_hits(args.input))
    if not hits:
        raise SystemExit(f"No hits found in {args.input}")

    x = np.array([hit.x for hit in hits], dtype=float)
    y = np.array([hit.y for hit in hits], dtype=float)
    toa = np.array([hit.toa for hit in hits], dtype=float)
    t = (toa - toa.min()) / 1_000_000.0

    features = np.column_stack([x, y, t / args.time_scale_m])
    labels = dbscan_labels(features, eps=args.eps, min_samples=args.min_samples)
    cluster_mask = labels != -1
    noise_mask = labels == -1
    n_clusters = len(set(int(label) for label in labels.tolist()) - {-1})

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x[noise_mask],
        y[noise_mask],
        t[noise_mask],
        c="lightgray",
        s=10,
        alpha=0.25,
        label="noise",
    )
    scatter = ax.scatter(
        x[cluster_mask],
        y[cluster_mask],
        t[cluster_mask],
        c=labels[cluster_mask],
        cmap="turbo",
        s=12,
        alpha=0.85,
    )

    ax.set_title(
        f"3D DBSCAN: {args.input.name}\n"
        f"clusters={n_clusters}, noise={int(noise_mask.sum())}, "
        f"eps={args.eps}, min_samples={args.min_samples}, time_scale={args.time_scale_m:g}M"
    )
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_zlabel("relative ToA (million ticks)")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="cluster id")
    plt.show()


def _import_pyplot(preferred_backend: str | None):
    """Import pyplot with a GUI backend, falling back when Qt/Tk is unavailable."""

    requested = []
    if preferred_backend:
        requested.append(preferred_backend)
    elif os.environ.get("MPLBACKEND"):
        requested.append(os.environ["MPLBACKEND"])

    requested.extend(["gtk4agg", "gtk3agg", "TkAgg", "QtAgg"])
    seen = set()
    errors = []
    for backend in requested:
        if backend in seen:
            continue
        seen.add(backend)
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt

            print(f"Using Matplotlib backend: {matplotlib.get_backend()}")
            return plt
        except ImportError as exc:
            errors.append(f"{backend}: {exc}")

    raise SystemExit("Could not load a Matplotlib GUI backend:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()

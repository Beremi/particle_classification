from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover - optional dependency
    DBSCAN = None

from particle_classification.clustering import dbscan_labels

from .models import DBSCANResult, MatrixRecord


DEFAULT_SET_INDICES = (1, 2, 3, 4, 5, 6)
DEFAULT_AXIS_TICKS = [0, 64, 128, 192, 255]


def to_dense_matrix(record: MatrixRecord) -> np.ndarray:
    matrix = np.zeros(record.shape, dtype=float)
    for entry in record.entries:
        matrix[entry.y, entry.x] = entry.energy
    return matrix


def plot_matrix(
    record: MatrixRecord,
    *,
    ax=None,
    log_scale: bool = True,
    origin: str = "lower",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    matrix = to_dense_matrix(record)
    masked = np.ma.masked_where(matrix <= 0, matrix)
    norm = _build_norm([matrix], log_scale=log_scale)
    image = ax.imshow(masked, origin=origin, cmap=_masked_cmap(), norm=norm)
    ax.set_title(f"sample={record.sample} set_index={record.set_index} nnz={record.nnz}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax, image


def plot_active_hits(record: MatrixRecord, *, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if not record.entries:
        ax.text(0.5, 0.5, "No active hits", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, record.shape[1] - 1)
        ax.set_ylim(0, record.shape[0] - 1)
        return ax, None

    xs = np.array([entry.x for entry in record.entries])
    ys = np.array([entry.y for entry in record.entries])
    energies = np.array([entry.energy for entry in record.entries], dtype=float)
    scatter = ax.scatter(
        xs,
        ys,
        c=energies,
        s=18,
        cmap="viridis",
        norm=colors.LogNorm(vmin=float(np.min(energies)), vmax=float(np.max(energies))),
        linewidths=0,
    )
    ax.set_xlim(0, record.shape[1] - 1)
    ax.set_ylim(0, record.shape[0] - 1)
    ax.set_aspect("equal")
    ax.set_title(f"Active hits: sample={record.sample} set_index={record.set_index}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax, scatter


def plot_sample_grid(
    records: Sequence[MatrixRecord],
    sample: int,
    *,
    layout: tuple[int, int] = (2, 3),
    set_indices: Sequence[int] = DEFAULT_SET_INDICES,
):
    sample_records = {record.set_index: record for record in records if record.sample == sample}
    if not sample_records:
        raise KeyError(f"Sample {sample} is not present in the records.")

    fig, axes = plt.subplots(*layout, figsize=(14, 8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    matrices = [to_dense_matrix(record) for record in sample_records.values()]
    norm = _build_norm(matrices, log_scale=True)
    cmap = _masked_cmap()
    images = []

    for ax, set_index in zip(axes, set_indices):
        _configure_matrix_axis(ax)
        record = sample_records.get(set_index)
        if record is None:
            ax.set_title(f"set_index={set_index}")
            _set_missing_axis_state(ax)
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            images.append(None)
            continue

        matrix = to_dense_matrix(record)
        masked = np.ma.masked_where(matrix <= 0, matrix)
        image = ax.imshow(masked, origin="lower", cmap=cmap, norm=norm)
        _set_present_axis_state(ax)
        ax.set_title(f"set_index={set_index}")
        images.append(image)

    title_record = next(iter(sample_records.values()))
    fig.suptitle(f"sample={sample} acq_unix={title_record.acq_unix:.1f}", fontsize=14)

    reference_image = next((image for image in images if image is not None), None)
    if reference_image is not None:
        fig.colorbar(reference_image, ax=axes.tolist(), shrink=0.92, pad=0.02, label="Energy")

    return fig, axes.reshape(layout)


def animate_samples(
    records: Sequence[MatrixRecord],
    *,
    samples: Sequence[int] | None = None,
    interval_ms: int = 500,
    save_path: str | Path | None = None,
):
    sample_to_records = _sample_lookup(records)
    ordered_samples = sorted(samples or sample_to_records.keys())
    if not ordered_samples:
        raise ValueError("No samples available for animation.")

    matrices = []
    for sample in ordered_samples:
        sample_records = sample_to_records[sample]
        for set_index in DEFAULT_SET_INDICES:
            record = sample_records.get(set_index)
            if record is not None:
                matrices.append(to_dense_matrix(record))

    norm = _build_norm(matrices, log_scale=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    cmap = _masked_cmap()
    images = []
    missing_labels = []

    initial_records = sample_to_records[ordered_samples[0]]
    for ax, set_index in zip(axes, DEFAULT_SET_INDICES):
        _configure_matrix_axis(ax)
        matrix = _matrix_or_zeros(initial_records.get(set_index))
        masked = np.ma.masked_where(matrix <= 0, matrix)
        image = ax.imshow(masked, origin="lower", cmap=cmap, norm=norm)
        label = ax.text(
            0.5,
            0.5,
            "",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="#555555",
        )
        if initial_records.get(set_index) is None:
            label.set_text("missing")
            _set_missing_axis_state(ax)
        else:
            _set_present_axis_state(ax)
        ax.set_title(f"set_index={set_index}")
        images.append(image)
        missing_labels.append(label)

    fig.colorbar(images[0], ax=axes.tolist(), shrink=0.92, pad=0.02, label="Energy")

    def update(frame_index: int):
        sample = ordered_samples[frame_index]
        sample_records = sample_to_records[sample]
        title_record = next(iter(sample_records.values()))
        fig.suptitle(f"sample={sample} acq_unix={title_record.acq_unix:.1f}", fontsize=14)

        artists = []
        for ax, image, label, set_index in zip(axes, images, missing_labels, DEFAULT_SET_INDICES):
            record = sample_records.get(set_index)
            matrix = _matrix_or_zeros(record)
            masked = np.ma.masked_where(matrix <= 0, matrix)
            image.set_data(masked)
            label.set_text("missing" if record is None else "")
            if record is None:
                _set_missing_axis_state(ax)
            else:
                _set_present_axis_state(ax)
            artists.extend([image, label])
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(ordered_samples),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, round(1000 / interval_ms))
        animation.save(output_path, writer=PillowWriter(fps=fps))

    return animation, fig


def animation_to_html(animation: FuncAnimation) -> HTML:
    return HTML(animation.to_jshtml())


def run_dbscan(
    record: MatrixRecord,
    *,
    eps: float = 1.5,
    min_samples: int = 3,
    feature_mode: str = "xy",
) -> DBSCANResult:
    if feature_mode != "xy":
        raise ValueError("V1 only supports feature_mode='xy'.")

    if not record.entries:
        empty = np.empty((0, 2), dtype=float)
        return DBSCANResult(
            features=empty,
            labels=np.empty((0,), dtype=int),
            xs=np.empty((0,), dtype=float),
            ys=np.empty((0,), dtype=float),
            energies=np.empty((0,), dtype=float),
            eps=eps,
            min_samples=min_samples,
            feature_mode=feature_mode,
        )

    xs = np.array([entry.x for entry in record.entries], dtype=float)
    ys = np.array([entry.y for entry in record.entries], dtype=float)
    energies = np.array([entry.energy for entry in record.entries], dtype=float)
    features = np.column_stack([xs, ys])
    if DBSCAN is None:
        labels = dbscan_labels(features, eps=eps, min_samples=min_samples)
    else:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)

    return DBSCANResult(
        features=features,
        labels=labels,
        xs=xs,
        ys=ys,
        energies=energies,
        eps=eps,
        min_samples=min_samples,
        feature_mode=feature_mode,
    )


def plot_dbscan_comparison(
    record: MatrixRecord,
    *,
    eps: float = 1.5,
    min_samples: int = 3,
):
    result = run_dbscan(record, eps=eps, min_samples=min_samples)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    raw_ax = axes[0]
    if result.energies.size:
        raw_scatter = raw_ax.scatter(
            result.xs,
            result.ys,
            c=result.energies,
            s=18,
            cmap="viridis",
            norm=colors.LogNorm(
                vmin=float(np.min(result.energies)),
                vmax=float(np.max(result.energies)),
            ),
            linewidths=0,
        )
        fig.colorbar(raw_scatter, ax=raw_ax, shrink=0.85, pad=0.02, label="Energy")
    else:
        raw_ax.text(0.5, 0.5, "No active hits", ha="center", va="center", transform=raw_ax.transAxes)
    raw_ax.set_title("Raw active hits")
    raw_ax.set_xlim(0, record.shape[1] - 1)
    raw_ax.set_ylim(0, record.shape[0] - 1)
    raw_ax.set_aspect("equal")
    raw_ax.set_xlabel("x")
    raw_ax.set_ylabel("y")

    cluster_ax = axes[1]
    if result.labels.size:
        cluster_colors = _cluster_colors(result.labels)
        cluster_ax.scatter(result.xs, result.ys, c=cluster_colors, s=18, linewidths=0)
    else:
        cluster_ax.text(
            0.5,
            0.5,
            "No active hits",
            ha="center",
            va="center",
            transform=cluster_ax.transAxes,
        )
    cluster_ax.set_title(
        "DBSCAN labels\n"
        f"clusters={result.n_clusters} noise={result.n_noise} "
        f"(eps={eps}, min_samples={min_samples})"
    )
    cluster_ax.set_xlim(0, record.shape[1] - 1)
    cluster_ax.set_ylim(0, record.shape[0] - 1)
    cluster_ax.set_aspect("equal")
    cluster_ax.set_xlabel("x")
    cluster_ax.set_ylabel("y")

    fig.suptitle(f"sample={record.sample} set_index={record.set_index}", fontsize=14)
    return fig, axes, result


def _sample_lookup(records: Iterable[MatrixRecord]) -> dict[int, dict[int, MatrixRecord]]:
    lookup: dict[int, dict[int, MatrixRecord]] = {}
    for record in records:
        lookup.setdefault(record.sample, {})[record.set_index] = record
    return lookup


def _build_norm(matrices: Sequence[np.ndarray], *, log_scale: bool) -> colors.Normalize | None:
    positives = [matrix[matrix > 0] for matrix in matrices]
    positives = [values for values in positives if values.size > 0]
    if not positives:
        return None

    vmin = float(min(values.min() for values in positives))
    vmax = float(max(values.max() for values in positives))
    if not log_scale or vmin <= 0:
        return colors.Normalize(vmin=vmin, vmax=vmax)
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def _masked_cmap():
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    return cmap


def _matrix_or_zeros(record: MatrixRecord | None) -> np.ndarray:
    if record is None:
        return np.zeros((256, 256), dtype=float)
    return to_dense_matrix(record)


def _cluster_colors(labels: np.ndarray):
    palette = plt.get_cmap("tab20")
    colors_out = []
    for label in labels:
        if label == -1:
            colors_out.append("#8a8a8a")
        else:
            colors_out.append(palette(label % palette.N))
    return colors_out


def _configure_matrix_axis(ax):
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xticks(DEFAULT_AXIS_TICKS)
    ax.set_yticks(DEFAULT_AXIS_TICKS)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _set_missing_axis_state(ax):
    ax.set_facecolor("#f7f7f7")
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=True,
        labelleft=True,
        bottom=True,
        left=True,
    )


def _set_present_axis_state(ax):
    ax.set_facecolor("white")
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=True,
        labelleft=True,
        bottom=True,
        left=True,
    )

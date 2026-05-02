from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParticleCandidate:
    """Variable-size particle candidate with columns `x, y, t, e`."""

    hits: np.ndarray
    candidate_id: str | None = None
    sample: int | None = None
    set_index: int | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hits = np.asarray(self.hits, dtype=float)
        if hits.ndim != 2 or hits.shape[1] != 4:
            raise ValueError("ParticleCandidate.hits must be a 2D array with columns x, y, t, e.")
        object.__setattr__(self, "hits", hits)

    @property
    def n_hits(self) -> int:
        return int(self.hits.shape[0])

    @property
    def xy(self) -> np.ndarray:
        return self.hits[:, :2]

    @property
    def time(self) -> np.ndarray:
        return self.hits[:, 2]

    @property
    def energy(self) -> np.ndarray:
        return self.hits[:, 3]


@dataclass(frozen=True)
class GeometryDescriptor:
    centroid_xy: tuple[float, float]
    time_min: float
    theta_xy: float
    q_theta: float
    phi_t: float
    eigenvalues_3d: tuple[float, float, float]
    eigenvalues_xy: tuple[float, float]
    scale_vector: tuple[float, float, float]
    total_energy: float
    max_energy: float
    mean_energy: float
    n_hits: int


def weighted_pca_geometry(
    candidate: ParticleCandidate,
    *,
    time_scale: float = 1.0,
    epsilon: float = 1e-9,
) -> GeometryDescriptor:
    """Compute XY pose metadata and class-relevant geometry descriptors.

    `theta_xy` is intentionally returned as metadata. Downstream descriptor
    vectors should use shape ratios, scale, density, energy, and time pitch,
    but not `theta_xy` itself.
    """

    hits = candidate.hits
    if hits.size == 0:
        return GeometryDescriptor(
            centroid_xy=(0.0, 0.0),
            time_min=0.0,
            theta_xy=0.0,
            q_theta=0.0,
            phi_t=0.0,
            eigenvalues_3d=(0.0, 0.0, 0.0),
            eigenvalues_xy=(0.0, 0.0),
            scale_vector=(0.0, 0.0, 0.0),
            total_energy=0.0,
            max_energy=0.0,
            mean_energy=0.0,
            n_hits=0,
        )

    xy = hits[:, :2]
    t = hits[:, 2]
    e = np.clip(hits[:, 3], a_min=0.0, a_max=None)
    weights = e + epsilon
    weight_sum = float(np.sum(weights))

    centroid_xy = np.sum(xy * weights[:, None], axis=0) / weight_sum
    time_min = float(np.min(t))
    xyz = np.column_stack(
        [
            xy[:, 0] - centroid_xy[0],
            xy[:, 1] - centroid_xy[1],
            (t - time_min) * time_scale,
        ]
    )
    xyz_center = np.sum(xyz * weights[:, None], axis=0) / weight_sum
    xyz_centered = xyz - xyz_center
    cov_3d = _weighted_covariance(xyz_centered, weights, weight_sum)
    evals_3d, evecs_3d = _sorted_eigh(cov_3d, width=3)

    xy_centered = xy - centroid_xy
    cov_xy = _weighted_covariance(xy_centered, weights, weight_sum)
    evals_xy, evecs_xy = _sorted_eigh(cov_xy, width=2)

    principal_xy = _orient_vector(evecs_xy[:, 0])
    theta_xy = float(np.arctan2(principal_xy[1], principal_xy[0]))
    q_theta = float((evals_xy[0] - evals_xy[1]) / (evals_xy[0] + evals_xy[1] + epsilon))

    principal_3d = evecs_3d[:, 0]
    if principal_3d[0] < 0 or (abs(principal_3d[0]) <= epsilon and principal_3d[1] < 0):
        principal_3d = -principal_3d
    phi_t = float(np.arctan2(principal_3d[2], np.linalg.norm(principal_3d[:2]) + epsilon))

    scales = np.sqrt(np.maximum(evals_3d, 0.0))
    total_energy = float(np.sum(e))
    return GeometryDescriptor(
        centroid_xy=(float(centroid_xy[0]), float(centroid_xy[1])),
        time_min=time_min,
        theta_xy=theta_xy,
        q_theta=q_theta,
        phi_t=phi_t,
        eigenvalues_3d=tuple(float(v) for v in evals_3d),
        eigenvalues_xy=tuple(float(v) for v in evals_xy),
        scale_vector=tuple(float(v) for v in scales),
        total_energy=total_energy,
        max_energy=float(np.max(e)) if e.size else 0.0,
        mean_energy=total_energy / float(e.size) if e.size else 0.0,
        n_hits=int(e.size),
    )


def rotate_xy(
    candidate: ParticleCandidate,
    angle: float,
    *,
    center: tuple[float, float] | None = None,
    candidate_id: str | None = None,
) -> ParticleCandidate:
    """Rotate only detector-plane coordinates; time and energy stay unchanged."""

    if center is None:
        center = weighted_pca_geometry(candidate).centroid_xy
    cx, cy = center
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    xy = candidate.hits[:, :2] - np.asarray([cx, cy])
    rotated = np.column_stack([cos_a * xy[:, 0] - sin_a * xy[:, 1], sin_a * xy[:, 0] + cos_a * xy[:, 1]])
    hits = candidate.hits.copy()
    hits[:, :2] = rotated + np.asarray([cx, cy])
    return ParticleCandidate(
        hits=hits,
        candidate_id=candidate_id or candidate.candidate_id,
        sample=candidate.sample,
        set_index=candidate.set_index,
        source_path=candidate.source_path,
        metadata={**candidate.metadata, "xy_rotation_applied": angle},
    )


def canonicalize_xy(candidate: ParticleCandidate, descriptor: GeometryDescriptor | None = None) -> ParticleCandidate:
    descriptor = descriptor or weighted_pca_geometry(candidate)
    return rotate_xy(candidate, -descriptor.theta_xy, center=descriptor.centroid_xy)


def energy_density_profiles(
    candidate: ParticleCandidate,
    *,
    descriptor: GeometryDescriptor | None = None,
    longitudinal_bins: int = 16,
    radial_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized energy profiles in the canonical detector-plane frame."""

    if candidate.n_hits == 0:
        return np.zeros(longitudinal_bins), np.zeros(radial_bins)

    descriptor = descriptor or weighted_pca_geometry(candidate)
    canonical = canonicalize_xy(candidate, descriptor)
    xy = canonical.hits[:, :2] - np.asarray(descriptor.centroid_xy)
    energy = np.clip(canonical.energy, a_min=0.0, a_max=None)
    total = float(np.sum(energy))
    if total <= 0:
        return np.zeros(longitudinal_bins), np.zeros(radial_bins)

    longitudinal = _weighted_histogram(xy[:, 0], energy, longitudinal_bins)
    radial = _weighted_histogram(np.abs(xy[:, 1]), energy, radial_bins, start_zero=True)
    return longitudinal / total, radial / total


def angle_distance_mod_pi(a: float, b: float) -> float:
    """Smallest angular distance for undirected axes where theta and theta + pi match."""

    return float(abs(np.arctan2(np.sin(2 * (a - b)), np.cos(2 * (a - b))) / 2.0))


def _weighted_covariance(values: np.ndarray, weights: np.ndarray, weight_sum: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros((values.shape[1], values.shape[1]), dtype=float)
    return (values * weights[:, None]).T @ values / weight_sum


def _sorted_eigh(matrix: np.ndarray, *, width: int) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.zeros(width), np.eye(width)
    evals, evecs = np.linalg.eigh(matrix)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    return evals, evecs


def _orient_vector(vector: np.ndarray, *, epsilon: float = 1e-9) -> np.ndarray:
    out = vector.copy()
    if out[0] < 0 or (abs(out[0]) <= epsilon and out[1] < 0):
        out = -out
    return out


def _weighted_histogram(values: np.ndarray, weights: np.ndarray, bins: int, *, start_zero: bool = False) -> np.ndarray:
    if bins <= 0:
        raise ValueError("Number of bins must be positive.")
    vmin = 0.0 if start_zero else float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        out = np.zeros(bins, dtype=float)
        out[0] = float(np.sum(weights))
        return out
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax), weights=weights)
    return hist.astype(float)

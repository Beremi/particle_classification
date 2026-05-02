from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import ParticleCandidate, energy_density_profiles, weighted_pca_geometry


@dataclass(frozen=True)
class DescriptorResult:
    vector: np.ndarray
    names: tuple[str, ...]
    metadata: dict[str, Any]


def xy_invariant_descriptor(
    candidate: ParticleCandidate,
    *,
    longitudinal_bins: int = 16,
    radial_bins: int = 8,
) -> DescriptorResult:
    """Build a deterministic class descriptor that deliberately excludes `theta_xy`."""

    geom = weighted_pca_geometry(candidate)
    longitudinal, radial = energy_density_profiles(
        candidate,
        descriptor=geom,
        longitudinal_bins=longitudinal_bins,
        radial_bins=radial_bins,
    )
    e = np.clip(candidate.energy, a_min=0.0, a_max=None)
    total = float(np.sum(e))
    max_e = float(np.max(e)) if e.size else 0.0
    mean_e = float(np.mean(e)) if e.size else 0.0
    std_e = float(np.std(e)) if e.size else 0.0
    concentration = max_e / (total + 1e-9)

    evals_3d = np.asarray(geom.eigenvalues_3d, dtype=float)
    evals_xy = np.asarray(geom.eigenvalues_xy, dtype=float)
    scale = np.asarray(geom.scale_vector, dtype=float)
    base = np.asarray(
        [
            np.log1p(geom.n_hits),
            np.log1p(total),
            np.log1p(max_e),
            np.log1p(mean_e),
            np.log1p(std_e),
            concentration,
            _ratio(evals_xy, 1, 0),
            _ratio(evals_3d, 1, 0),
            _ratio(evals_3d, 2, 0),
            np.log1p(scale[0]),
            np.log1p(scale[1]),
            np.log1p(scale[2]),
            np.cos(geom.phi_t),
            np.sin(geom.phi_t),
        ],
        dtype=float,
    )
    names = (
        "log_n_hits",
        "log_total_energy",
        "log_max_energy",
        "log_mean_energy",
        "log_std_energy",
        "energy_concentration",
        "xy_minor_major_ratio",
        "eig2_eig1_ratio",
        "eig3_eig1_ratio",
        "log_scale_1",
        "log_scale_2",
        "log_scale_3",
        "cos_phi_t",
        "sin_phi_t",
    )
    long_names = tuple(f"longitudinal_energy_bin_{i:02d}" for i in range(longitudinal_bins))
    radial_names = tuple(f"radial_energy_bin_{i:02d}" for i in range(radial_bins))
    vector = np.concatenate([base, longitudinal, radial])
    metadata = {
        "theta_xy": geom.theta_xy,
        "q_theta": geom.q_theta,
        "phi_t": geom.phi_t,
        "total_energy": geom.total_energy,
        "max_energy": geom.max_energy,
        "mean_energy": geom.mean_energy,
        "n_hits": geom.n_hits,
    }
    return DescriptorResult(vector=vector, names=names + long_names + radial_names, metadata=metadata)


def stack_descriptors(candidates: list[ParticleCandidate]) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, Any]]]:
    results = [xy_invariant_descriptor(candidate) for candidate in candidates]
    if not results:
        return np.empty((0, 0), dtype=float), tuple(), []
    return (
        np.vstack([result.vector for result in results]),
        results[0].names,
        [result.metadata for result in results],
    )


def _ratio(values: np.ndarray, numerator: int, denominator: int) -> float:
    if values.size <= max(numerator, denominator):
        return 0.0
    return float(values[numerator] / (values[denominator] + 1e-9))

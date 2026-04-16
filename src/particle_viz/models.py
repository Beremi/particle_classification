from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatrixEntry:
    x: int
    y: int
    energy: float


@dataclass(frozen=True)
class MatrixRecord:
    sample: int
    set_index: int
    acq_unix: float
    hw_t0_ns: float
    hw_t_proc_ns: float
    shape: tuple[int, int]
    nnz: int
    entries: tuple[MatrixEntry, ...]


@dataclass(frozen=True)
class DBSCANResult:
    features: np.ndarray
    labels: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    energies: np.ndarray
    eps: float
    min_samples: int
    feature_mode: str

    @property
    def n_clusters(self) -> int:
        unique = set(self.labels.tolist())
        return len([label for label in unique if label != -1])

    @property
    def n_noise(self) -> int:
        return int(np.sum(self.labels == -1))

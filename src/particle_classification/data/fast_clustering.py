from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class BackendUnavailable(RuntimeError):
    """Raised when an optional accelerated clustering backend is not installed."""


@dataclass(frozen=True)
class GridOffsets:
    dx: np.ndarray
    dy: np.ndarray
    dt: np.ndarray
    spatial2: np.ndarray


def native_grid_dbscan_labels(
    x: np.ndarray,
    y: np.ndarray,
    time_scaled: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    threads: int = 0,
) -> np.ndarray:
    native = _load_native()
    x_arr, y_arr, t_arr, order = _prepare_grid_inputs(x, y, time_scaled)
    return native.grid_dbscan_ordered(x_arr, y_arr, t_arr, order, float(eps), int(min_samples), int(threads))


def native_stream_grid_linker_labels(
    x: np.ndarray,
    y: np.ndarray,
    time_scaled: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    threads: int = 0,
) -> np.ndarray:
    native = _load_native()
    x_arr, y_arr, t_arr, order = _prepare_grid_inputs(x, y, time_scaled)
    return native.stream_grid_linker_ordered(x_arr, y_arr, t_arr, order, float(eps), int(min_samples), int(threads))


def native_voxel_connected_components_labels(
    x: np.ndarray,
    y: np.ndarray,
    time_bin: np.ndarray,
    *,
    connectivity: str,
    min_hits: int,
    threads: int = 0,
) -> np.ndarray:
    native = _load_native()
    x_arr, y_arr, t_arr = _prepare_voxel_inputs(x, y, time_bin)
    connectivity_code = {"face": 1, "edge": 2, "corner": 3}[connectivity]
    return native.voxel_connected_components(
        x_arr,
        y_arr,
        t_arr,
        connectivity_code,
        int(max(1, min_hits)),
        int(threads),
    )


def numba_grid_dbscan_labels(
    x: np.ndarray,
    y: np.ndarray,
    time_scaled: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    threads: int = 0,
) -> np.ndarray:
    try:
        from .numba_clustering import grid_dbscan_ordered_numba, set_numba_threads
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise BackendUnavailable("Install `particle-classification[speed]` to use Numba backends.") from exc

    x_arr, y_arr, t_arr, order = _prepare_grid_inputs(x, y, time_scaled)
    offsets = make_grid_offsets(eps)
    set_numba_threads(threads)
    return grid_dbscan_ordered_numba(
        x_arr,
        y_arr,
        t_arr,
        order,
        offsets.dx,
        offsets.dy,
        offsets.dt,
        offsets.spatial2,
        float(eps * eps),
        int(min_samples),
    )


def numba_stream_grid_linker_labels(
    x: np.ndarray,
    y: np.ndarray,
    time_scaled: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    threads: int = 0,
) -> np.ndarray:
    try:
        from .numba_clustering import set_numba_threads, stream_grid_linker_ordered_numba
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise BackendUnavailable("Install `particle-classification[speed]` to use Numba backends.") from exc

    x_arr, y_arr, t_arr, order = _prepare_grid_inputs(x, y, time_scaled)
    offsets = make_grid_offsets(eps)
    set_numba_threads(threads)
    return stream_grid_linker_ordered_numba(
        x_arr,
        y_arr,
        t_arr,
        order,
        offsets.dx,
        offsets.dy,
        offsets.dt,
        offsets.spatial2,
        float(eps * eps),
        int(min_samples),
    )


def is_backend_available(backend: str) -> bool:
    if backend.startswith("native-"):
        try:
            _load_native()
            return True
        except BackendUnavailable:
            return False
    if backend.startswith("numba-"):
        try:
            import numba  # noqa: F401

            return True
        except Exception:
            return False
    return True


def make_grid_offsets(eps: float) -> GridOffsets:
    eps = float(eps)
    eps2 = eps * eps
    radius = int(math.ceil(eps))
    dxs: list[int] = []
    dys: list[int] = []
    dts: list[float] = []
    spatial: list[float] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial2 = float(dx * dx + dy * dy)
            if spatial2 <= eps2 + 1e-12:
                dxs.append(dx)
                dys.append(dy)
                spatial.append(spatial2)
                dts.append(math.sqrt(max(0.0, eps2 - spatial2)))
    return GridOffsets(
        dx=np.asarray(dxs, dtype=np.int16),
        dy=np.asarray(dys, dtype=np.int16),
        dt=np.asarray(dts, dtype=np.float64),
        spatial2=np.asarray(spatial, dtype=np.float64),
    )


def _prepare_grid_inputs(
    x: np.ndarray,
    y: np.ndarray,
    time_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.ascontiguousarray(x, dtype=np.uint16)
    y_arr = np.ascontiguousarray(y, dtype=np.uint16)
    t_arr = np.ascontiguousarray(time_scaled, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or t_arr.ndim != 1:
        raise ValueError("Grid clustering inputs must be one-dimensional arrays.")
    if x_arr.shape[0] != y_arr.shape[0] or x_arr.shape[0] != t_arr.shape[0]:
        raise ValueError("Grid clustering inputs must have matching lengths.")
    if x_arr.size and (int(np.max(x_arr)) > 255 or int(np.max(y_arr)) > 255):
        raise ValueError("Grid clustering requires detector coordinates in the 0..255 range.")
    order = np.ascontiguousarray(np.argsort(t_arr, kind="mergesort"), dtype=np.int64)
    return x_arr, y_arr, t_arr, order


def _prepare_voxel_inputs(
    x: np.ndarray,
    y: np.ndarray,
    time_bin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.ascontiguousarray(x, dtype=np.uint16)
    y_arr = np.ascontiguousarray(y, dtype=np.uint16)
    t_arr = np.ascontiguousarray(time_bin, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or t_arr.ndim != 1:
        raise ValueError("Voxel clustering inputs must be one-dimensional arrays.")
    if x_arr.shape[0] != y_arr.shape[0] or x_arr.shape[0] != t_arr.shape[0]:
        raise ValueError("Voxel clustering inputs must have matching lengths.")
    if x_arr.size and (int(np.max(x_arr)) > 255 or int(np.max(y_arr)) > 255):
        raise ValueError("Voxel clustering requires detector coordinates in the 0..255 range.")
    return x_arr, y_arr, t_arr


def _load_native():
    try:
        from particle_classification import _native_clustering
    except Exception as exc:  # pragma: no cover - depends on local build
        raise BackendUnavailable(
            "Native clustering extension is not built. Run `python setup.py build_ext --inplace` "
            "or reinstall the package."
        ) from exc
    return _native_clustering

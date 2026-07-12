from __future__ import annotations

import numpy as np
from numba import njit, prange, set_num_threads


def set_numba_threads(threads: int) -> None:
    if threads and threads > 0:
        set_num_threads(int(threads))


@njit(cache=True)
def _lower_bound_bucket(indices, start, end, t, value):
    lo = start
    hi = end
    while lo < hi:
        mid = (lo + hi) // 2
        if t[indices[mid]] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _upper_bound_bucket(indices, start, end, t, value):
    lo = start
    hi = end
    while lo < hi:
        mid = (lo + hi) // 2
        if t[indices[mid]] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _find(parent, idx):
    root = idx
    while parent[root] != root:
        root = parent[root]
    while parent[idx] != idx:
        nxt = parent[idx]
        parent[idx] = root
        idx = nxt
    return root


@njit(cache=True)
def _union(parent, rank, a, b):
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a == root_b:
        return root_a
    if rank[root_a] < rank[root_b]:
        tmp = root_a
        root_a = root_b
        root_b = tmp
    parent[root_b] = root_a
    if rank[root_a] == rank[root_b]:
        rank[root_a] += 1
    return root_a


@njit(cache=True)
def _build_sorted_pixel_buckets(x, y, order):
    n = x.shape[0]
    counts = np.zeros(65536, dtype=np.int64)
    for i in range(n):
        pixel = int(y[i]) * 256 + int(x[i])
        counts[pixel] += 1
    offsets = np.zeros(65537, dtype=np.int64)
    for pixel in range(65536):
        offsets[pixel + 1] = offsets[pixel] + counts[pixel]
    cursor = offsets[:-1].copy()
    bucket_indices = np.empty(n, dtype=np.int64)
    for pos in range(n):
        idx = int(order[pos])
        pixel = int(y[idx]) * 256 + int(x[idx])
        out_pos = cursor[pixel]
        bucket_indices[out_pos] = idx
        cursor[pixel] += 1
    return offsets, bucket_indices


@njit(cache=True, parallel=True)
def _mark_core_points(x, y, t, offsets, bucket_indices, dxs, dys, dts, spatial2, eps2, min_samples):
    n = x.shape[0]
    core = np.zeros(n, dtype=np.uint8)
    tol = 1e-12
    for i in prange(n):
        xi = int(x[i])
        yi = int(y[i])
        ti = t[i]
        count = 1
        for o in range(dxs.shape[0]):
            nx = xi + int(dxs[o])
            ny = yi + int(dys[o])
            if nx < 0 or nx >= 256 or ny < 0 or ny >= 256:
                continue
            pixel = ny * 256 + nx
            start = offsets[pixel]
            end = offsets[pixel + 1]
            if start == end:
                continue
            lo = _lower_bound_bucket(bucket_indices, start, end, t, ti - dts[o] - tol)
            hi = _upper_bound_bucket(bucket_indices, start, end, t, ti + dts[o] + tol)
            for pos in range(lo, hi):
                j = int(bucket_indices[pos])
                if j == i:
                    continue
                dt = t[j] - ti
                if spatial2[o] + dt * dt <= eps2 + tol:
                    count += 1
                    if count >= min_samples:
                        core[i] = 1
                        break
            if core[i] == 1:
                break
    return core


@njit(cache=True)
def grid_dbscan_ordered_numba(x, y, t, order, dxs, dys, dts, spatial2, eps2, min_samples):
    n = x.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    offsets, bucket_indices = _build_sorted_pixel_buckets(x, y, order)
    core = _mark_core_points(x, y, t, offsets, bucket_indices, dxs, dys, dts, spatial2, eps2, min_samples)
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)
    tol = 1e-12

    for i in range(n):
        if core[i] == 0:
            continue
        xi = int(x[i])
        yi = int(y[i])
        ti = t[i]
        for o in range(dxs.shape[0]):
            nx = xi + int(dxs[o])
            ny = yi + int(dys[o])
            if nx < 0 or nx >= 256 or ny < 0 or ny >= 256:
                continue
            pixel = ny * 256 + nx
            lo = _lower_bound_bucket(bucket_indices, offsets[pixel], offsets[pixel + 1], t, ti - dts[o] - tol)
            hi = _upper_bound_bucket(bucket_indices, offsets[pixel], offsets[pixel + 1], t, ti + dts[o] + tol)
            for pos in range(lo, hi):
                j = int(bucket_indices[pos])
                if j <= i or core[j] == 0:
                    continue
                dt = t[j] - ti
                if spatial2[o] + dt * dt <= eps2 + tol:
                    _union(parent, rank, i, j)

    root_label = np.full(n, -1, dtype=np.int32)
    next_label = 0
    for i in range(n):
        if core[i] != 0:
            root = _find(parent, i)
            if root_label[root] < 0:
                root_label[root] = next_label
                next_label += 1
            labels[i] = root_label[root]

    for i in range(n):
        if core[i] != 0:
            continue
        xi = int(x[i])
        yi = int(y[i])
        ti = t[i]
        done = False
        for o in range(dxs.shape[0]):
            if done:
                break
            nx = xi + int(dxs[o])
            ny = yi + int(dys[o])
            if nx < 0 or nx >= 256 or ny < 0 or ny >= 256:
                continue
            pixel = ny * 256 + nx
            lo = _lower_bound_bucket(bucket_indices, offsets[pixel], offsets[pixel + 1], t, ti - dts[o] - tol)
            hi = _upper_bound_bucket(bucket_indices, offsets[pixel], offsets[pixel + 1], t, ti + dts[o] + tol)
            for pos in range(lo, hi):
                j = int(bucket_indices[pos])
                if core[j] == 0:
                    continue
                dt = t[j] - ti
                if spatial2[o] + dt * dt <= eps2 + tol:
                    labels[i] = labels[j]
                    done = True
                    break
    return labels


@njit(cache=True)
def stream_grid_linker_ordered_numba(x, y, t, order, dxs, dys, dts, spatial2, eps2, min_samples):
    n = x.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    offsets, bucket_indices = _build_sorted_pixel_buckets(x, y, order)
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)
    tol = 1e-12

    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        ti = t[i]
        for o in range(dxs.shape[0]):
            nx = xi + int(dxs[o])
            ny = yi + int(dys[o])
            if nx < 0 or nx >= 256 or ny < 0 or ny >= 256:
                continue
            pixel = ny * 256 + nx
            lo = _lower_bound_bucket(bucket_indices, offsets[pixel], offsets[pixel + 1], t, ti - dts[o] - tol)
            hi = _upper_bound_bucket(bucket_indices, offsets[pixel], offsets[pixel + 1], t, ti + dts[o] + tol)
            for pos in range(lo, hi):
                j = int(bucket_indices[pos])
                if j <= i:
                    continue
                dt = t[j] - ti
                if spatial2[o] + dt * dt <= eps2 + tol:
                    _union(parent, rank, i, j)

    sizes = np.zeros(n, dtype=np.int32)
    for i in range(n):
        root = _find(parent, i)
        sizes[root] += 1
    root_label = np.full(n, -1, dtype=np.int32)
    next_label = 0
    for i in range(n):
        root = _find(parent, i)
        if sizes[root] < min_samples:
            continue
        if root_label[root] < 0:
            root_label[root] = next_label
            next_label += 1
        labels[i] = root_label[root]
    return labels

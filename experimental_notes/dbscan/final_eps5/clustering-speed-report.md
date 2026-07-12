# 3D Clustering Speed Report

Status: supports the active Phase 1 baseline. `native-grid-dbscan` is the
recommended separator backend for Phase 1 particle extraction.

This report compares generic SciPy DBSCAN paths, optional Numba grid kernels, and native C/OpenMP grid kernels for Timepix 3D particle separation.

## Throughput and RT Ratio

| Case | Backend | Kind | Hits | Runtime s | Hits/s | RT ratio | Speedup vs `ckdtree-pairs` | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| largest | `ckdtree-pairs` | exact DBSCAN | 6275049 | 219.761 | 28554.0 | 3.68x | 1.00x | ok |
| largest | `numba-grid-dbscan` | exact DBSCAN | 6275049 | 24.888 | 252131.5 | 0.42x | 8.83x | ok |
| largest | `native-grid-dbscan` | exact DBSCAN | 6275049 | 24.134 | 260008.7 | 0.40x | 9.11x | ok |
| largest | `numba-stream-grid-linker` | stream linker | 6275049 | 47.135 | 133129.3 | 0.79x | 4.66x | ok |
| largest | `native-stream-grid-linker` | stream linker | 6275049 | 40.953 | 153225.6 | 0.68x | 5.37x | ok |
| slowest_per_hit | `ckdtree-pairs` | exact DBSCAN | 2452962 | 109.285 | 22445.6 | 5.47x | 1.00x | ok |
| slowest_per_hit | `numba-grid-dbscan` | exact DBSCAN | 2452962 | 8.580 | 285893.0 | 0.43x | 12.74x | ok |
| slowest_per_hit | `native-grid-dbscan` | exact DBSCAN | 2452962 | 8.285 | 296072.7 | 0.41x | 13.19x | ok |
| slowest_per_hit | `numba-stream-grid-linker` | stream linker | 2452962 | 13.950 | 175839.6 | 0.70x | 7.83x | ok |
| slowest_per_hit | `native-stream-grid-linker` | stream linker | 2452962 | 13.080 | 187535.3 | 0.65x | 8.36x | ok |
| max_particles | `ckdtree-pairs` | exact DBSCAN | 21657 | 0.080 | 270712.5 | 0.00x | 1.00x | ok |
| max_particles | `numba-grid-dbscan` | exact DBSCAN | 21657 | 0.055 | 393763.6 | 0.00x | 1.45x | ok |
| max_particles | `native-grid-dbscan` | exact DBSCAN | 21657 | 0.060 | 360950.0 | 0.00x | 1.33x | ok |
| max_particles | `numba-stream-grid-linker` | stream linker | 21657 | 0.054 | 401055.6 | 0.00x | 1.48x | ok |
| max_particles | `native-stream-grid-linker` | stream linker | 21657 | 0.056 | 386732.1 | 0.00x | 1.43x | ok |

## Largest File RT Verdict

| Backend | Runtime s | Physical length s | RT ratio | Faster than real time? |
|---|---:|---:|---:|---|
| `ckdtree-pairs` | 219.761 | 59.789 | 3.68x | no |
| `numba-grid-dbscan` | 24.888 | 59.789 | 0.42x | yes |
| `native-grid-dbscan` | 24.134 | 59.789 | 0.40x | yes |
| `numba-stream-grid-linker` | 47.135 | 59.789 | 0.79x | yes |
| `native-stream-grid-linker` | 40.953 | 59.789 | 0.68x | yes |

## Exact DBSCAN Label Agreement

| Case | Backend | ARI | Same noise fraction |
|---|---|---:|---:|
| largest | `ckdtree-pairs` | 1.000000 | 1.000000 |
| largest | `numba-grid-dbscan` | 1.000000 | 1.000000 |
| largest | `native-grid-dbscan` | 1.000000 | 1.000000 |
| slowest_per_hit | `ckdtree-pairs` | 1.000000 | 1.000000 |
| slowest_per_hit | `numba-grid-dbscan` | 1.000000 | 1.000000 |
| slowest_per_hit | `native-grid-dbscan` | 1.000000 | 1.000000 |
| max_particles | `ckdtree-pairs` | 1.000000 | 1.000000 |
| max_particles | `numba-grid-dbscan` | 1.000000 | 1.000000 |
| max_particles | `native-grid-dbscan` | 1.000000 | 1.000000 |

## Stream Linker Agreement

Stream-grid linkers are trajectory-oriented prototypes. They are compared with DBSCAN labels for orientation, but exact equality is not required.

| Case | Backend | ARI | Same noise fraction |
|---|---|---:|---:|
| largest | `numba-stream-grid-linker` | 1.000000 | 1.000000 |
| largest | `native-stream-grid-linker` | 1.000000 | 1.000000 |
| slowest_per_hit | `numba-stream-grid-linker` | 1.000000 | 1.000000 |
| slowest_per_hit | `native-stream-grid-linker` | 1.000000 | 1.000000 |
| max_particles | `numba-stream-grid-linker` | 1.000000 | 1.000000 |
| max_particles | `native-stream-grid-linker` | 1.000000 | 1.000000 |

## Summary

Exact DBSCAN backends are judged by agreement with the current `ckdtree-pairs` reference. Stream linkers are judged separately because they deliberately optimize for online trajectory grouping rather than DBSCAN semantics.

```json
{
  "by_backend": {
    "ckdtree-pairs": {
      "cases": 3,
      "hits_per_s": 26584.554243663522,
      "total_hits": 8749668,
      "total_runtime_s": 329.126
    },
    "native-grid-dbscan": {
      "cases": 3,
      "hits_per_s": 269394.62421872595,
      "total_hits": 8749668,
      "total_runtime_s": 32.479
    },
    "native-stream-grid-linker": {
      "cases": 3,
      "hits_per_s": 161764.27739466433,
      "total_hits": 8749668,
      "total_runtime_s": 54.089000000000006
    },
    "numba-grid-dbscan": {
      "cases": 3,
      "hits_per_s": 261004.92199385495,
      "total_hits": 8749668,
      "total_runtime_s": 33.523
    },
    "numba-stream-grid-linker": {
      "cases": 3,
      "hits_per_s": 143111.07476406224,
      "total_hits": 8749668,
      "total_runtime_s": 61.138999999999996
    }
  },
  "ckdtree_pairs_min_ari": 1.0,
  "exact_min_ari": {
    "ckdtree-pairs": 1.0,
    "native-grid-dbscan": 1.0,
    "numba-grid-dbscan": 1.0
  }
}
```

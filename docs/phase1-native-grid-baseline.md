# Phase 1 Baseline: Native Grid DBSCAN

Status: active Phase 1 baseline as of 2026-05-03.

The Phase 1 separator baseline is now `native-grid-dbscan`: a custom C/OpenMP
implementation of exact 3D DBSCAN over Timepix hit streams. It replaces the
earlier neural-network replacement attempts as the practical first-stage
particle separator.

## What It Does

Input hits are still the raw detector sequence:

```text
(x, y, ToA, ToT, FToA, source_row)
```

Clustering distance uses only:

```text
(x, y, ToA / time_scale)
```

Energy-like data are preserved in output shards as `hit_energy = log1p(ToT)`,
but energy is not part of the DBSCAN distance.

The backend exploits fixed detector geometry:

- `x,y` live on a fixed `256 x 256` integer pixel grid;
- valid `(dx, dy, dt_max)` neighbor offsets are precomputed from `eps`;
- core-point detection and radius checks use pixel buckets instead of a generic
  KD-tree;
- core clusters are joined with a custom union-find;
- border hits are assigned to neighboring core components with DBSCAN semantics;
- OpenMP parallelizes the core-point marking pass.

This is a custom implementation. It does not call scikit-learn DBSCAN, SciPy
KDTree, HDBSCAN, or any external clustering library for the native result.

## Parameters

The active teacher/baseline is `configs/teachers/dbscan_v001.json`:

| Parameter | Value |
|---|---:|
| `eps` | 5.0 |
| `min_samples` | 3 |
| `time_scale` | 15,000,000 |
| `window_size` | 250,000 |
| `window_overlap` | 25,000 |

Large files still use the existing overlapping time-window strategy for exact
DBSCAN compatibility. The native backend is used inside each window.

## Full-Data Validation

Validation reran `native-grid-dbscan` from raw `.t3pa` files and compared every
per-hit label with the existing `dbscan_v001` particle shards.

| Metric | Result |
|---|---:|
| Files validated | 711 / 711 |
| Hits validated | 44,725,206 / 44,725,206 |
| Files with exact label match | 711 / 711 |
| Hits with exact label match | 44,725,206 / 44,725,206 |
| Errors | 0 |
| Min ARI | 1.0 |
| Mean ARI | 1.0 |
| Min same-noise fraction | 1.0 |
| Total validation wall time | 161.94 s |
| Aggregate throughput including comparison | 276,188 hits/s |

By folder:

| Folder | Files | Hits | Exact Matches |
|---|---:|---:|---:|
| `08_thu_proton_daily_batch` | 368 | 1,390,014 | 368 |
| `13_tue_proton_daily_batch` | 229 | 1,757,135 | 229 |
| `D05` | 54 | 8,570,737 | 54 |
| `F08` | 54 | 15,474,240 | 54 |
| `data I05` | 3 | 13,947,284 | 3 |
| `data M07` | 3 | 3,585,796 | 3 |

Local validation artifacts:

```text
local_data/benchmarks/native_grid_all_validation/validation.csv
local_data/benchmarks/native_grid_all_validation/summary.json
```

## Speed Baseline

The largest file is `data I05/sync__I05-W0044_r001.t3pa` with 6,275,049 hits.
Assuming a 25 ns ToA tick, its physical span is 59.789 s.

| Backend | Runtime s | RT ratio | Label agreement |
|---|---:|---:|---:|
| `ckdtree-pairs` | 219.761 | 3.68x slower than RT | ARI 1.0 |
| `numba-grid-dbscan` | 24.888 | 0.42x RT | ARI 1.0 |
| `native-grid-dbscan` | 24.134 | 0.40x RT | ARI 1.0 |
| `numba-stream-grid-linker` | 47.135 | 0.79x RT | ARI 1.0 on benchmark cases |
| `native-stream-grid-linker` | 40.953 | 0.68x RT | ARI 1.0 on benchmark cases |

`native-grid-dbscan` is therefore about 2.5x faster than real time on the
largest tested file, including `.t3pa` parsing and label generation. Benchmark
mode does not include compressed NPZ writing.

## Commands

Build particle shards with the active baseline:

```bash
particle-build-particles \
  --input local_data/raw \
  --index data/raw_data_index.csv \
  --params configs/teachers/dbscan_v001.json \
  --out local_data/processed/particles_native_grid_v001 \
  --backend native-grid-dbscan \
  --threads 32
```

Benchmark the CPU backends:

```bash
OMP_NUM_THREADS=32 particle-benchmark-clustering \
  --input local_data/raw \
  --params configs/teachers/dbscan_v001.json \
  --out local_data/benchmarks/clustering_native_v001 \
  --cases largest,slowest_per_hit,max_particles \
  --backend ckdtree-pairs \
  --backend numba-grid-dbscan \
  --backend native-grid-dbscan \
  --backend numba-stream-grid-linker \
  --backend native-stream-grid-linker \
  --threads 32 \
  --repeat-runs 1 \
  --toa-tick-ns 25
```

## Relationship To Neural Attempts

The neural Phase 1 work is archived as research material. It remains useful for
future learned clustering and classification experiments, but it is not the
current Phase 1 separator baseline.

Reasons:

- `native-grid-dbscan` exactly reproduces the current teacher labels across all
  available local data.
- It is already faster than real time on the largest file.
- The NN attempts were promising on stable pseudo-label windows but did not
  reliably solve hard mixed overlap cases.

The next implementation phase should build classification and testing workflows
on top of native-generated particle shards, while keeping the NN reports as
background for future learned alternatives.

# Phase 1 Baseline: Native Grid DBSCAN

Status: final Phase 1 separator baseline as of 2026-05-05.

The Phase 1 particle separator is `native-grid-dbscan`: a custom C/OpenMP
implementation of exact 3D DBSCAN over Timepix hit streams. Neural separator
attempts remain archived as research notes, but the implementation baseline is
now the native DBSCAN path with corrected fine timestamp handling.

The frozen teacher config is:

```text
configs/teachers/dbscan_phase1_native_eps5_v001.json
```

## Input And Distance

Input hits are raw detector rows:

```text
(x, y, ToA, ToT, FToA, source_row)
```

The parser now reconstructs fine time as:

```text
hit_time = ToA - FToA / 16
```

DBSCAN distance uses only:

```text
(x, y, hit_time / time_scale)
```

Energy-like data are preserved in output shards as `hit_energy = log1p(ToT)`,
but energy is not part of DBSCAN distance.

## Backend

The native backend exploits fixed detector geometry:

- `x,y` live on a fixed `256 x 256` integer pixel grid;
- valid `(dx, dy, dt_max)` neighbor offsets are precomputed from `eps`;
- core-point detection and radius checks use pixel buckets instead of a generic
  KD-tree;
- core clusters are joined with a custom union-find;
- border hits are assigned to neighboring core components with DBSCAN semantics;
- OpenMP parallelizes the core-point marking pass.

This is a custom implementation. It does not call scikit-learn DBSCAN, SciPy
KDTree, HDBSCAN, or any external clustering library for the native result.

## Final Parameters

| Parameter | Value |
|---|---:|
| `eps` | 5.0 |
| `min_samples` | 2 |
| `time_scale` | 0.625 |
| `window_size` | 250,000 |
| `window_overlap` | 25,000 |

In physical units, `time_scale = 0.625` ToA ticks is `15.625 ns`. A unit in the
DBSCAN time coordinate is therefore 10 fine FToA bins.

Large files use overlapping time windows with union-find merging across overlap
hits. The native backend runs inside each window.

## Full Reestimate

The final pass rebuilt all local `.t3pa` files into:

```text
local_data/processed/particles_aligned_time_eps5_v001/
```

| Metric | Result |
|---|---:|
| Files processed | 711 / 711 |
| Raw hits | 44,725,206 |
| Estimated particles | 3,464,792 |
| Weighted noise fraction | 0.024 |
| Slowest file runtime | 36.063 s |
| Largest file runtime | 36.063 s |

The final inspection reports are:

- [DBSCAN eps5 realignment](dbscan-aligned-time-eps5-realignment.md)
- [Approx. 100-particle windows](native-grid-dbscan-aligned-time-eps5-approx-100-particle-windows.md)
- [Longest-timespan particles](native-grid-dbscan-aligned-time-eps5-long-timespan-particles.md)

## Real-Time Check

The table uses physical duration from corrected `hit_time` span with a 25 ns ToA
tick. `RT fraction = runtime / physical duration`; values below 1.0 are faster
than real time.

| Case | File | Hits | Duration s | Runtime s | RT fraction | Speed |
|---|---|---:|---:|---:|---:|---:|
| Densest hit-rate file | `data I05/sync__I05-W0044_r002.t3pa` | 2,156,669 | 15.632 | 12.209 | 0.781 | 1.28x RT |
| Largest file | `data I05/sync__I05-W0044_r001.t3pa` | 6,275,049 | 59.789 | 36.063 | 0.603 | 1.66x RT |
| Slow dense F08 file | `F08/tot_toa__r0000000052.t3pa` | 2,452,962 | 19.996 | 13.160 | 0.658 | 1.52x RT |

This runtime includes file processing and NPZ shard writing as recorded in the
particle manifest. The densest case is real-time capable, but with only about
28% headroom; a streaming deployment should avoid compressed NPZ writing on the
hot path.

## Historical Audit

The old `dbscan_v001` config remains useful for backend agreement testing, but
not for physical particle labels. A later audit found file-scale merged
components under those parameters. See [dbscan-teacher-audit.md](dbscan-teacher-audit.md).

## Commands

Build particle shards with the final Phase 1 baseline:

```bash
OMP_NUM_THREADS=32 particle-build-particles \
  --input local_data/raw \
  --index data/raw_data_index.csv \
  --params configs/teachers/dbscan_phase1_native_eps5_v001.json \
  --out local_data/processed/particles_aligned_time_eps5_v001 \
  --backend native-grid-dbscan \
  --threads 32 \
  --skip-existing
```

Generate the long-timespan inspection gallery:

```bash
.venv/bin/python scripts/render_long_timespan_particles.py \
  --manifest local_data/processed/particles_aligned_time_eps5_v001/manifest.csv \
  --assets docs/assets/native_grid_dbscan_aligned_time_eps5_long_timespan_particles \
  --report docs/native-grid-dbscan-aligned-time-eps5-long-timespan-particles.md \
  --limit 8 \
  --time-scale 0.625
```

Generate approx. 100-particle inspection windows:

```bash
.venv/bin/python scripts/render_approx_particle_window_gallery.py \
  --manifest local_data/processed/particles_aligned_time_eps5_v001/manifest.csv \
  --assets docs/assets/native_grid_dbscan_aligned_time_eps5_approx_100_particle_windows \
  --report docs/native-grid-dbscan-aligned-time-eps5-approx-100-particle-windows.md \
  --target-particles 100 \
  --tolerance 25 \
  --windows 8 \
  --time-scale 0.625
```

## Relationship To Neural Attempts

The neural Phase 1 work is archived as research material. It remains useful for
future learned clustering and classification experiments, but it is not the
current Phase 1 separator baseline.

Reasons:

- the native implementation is real-time capable on the densest local file;
- visual checks on crowded windows and long-span tracks look physically healthy
  after fine-time correction and eps5 realignment;
- the NN attempts were promising on stable pseudo-label windows but did not
  reliably solve hard mixed overlap cases.

The next implementation phase should build classification and testing workflows
on top of `particles_aligned_time_eps5_v001` or regenerated shards from the same
teacher config.

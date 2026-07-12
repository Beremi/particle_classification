# Native grid DBSCAN

The active Phase 1 separator is `native-grid-dbscan`, a custom exact DBSCAN
implementation specialized for Timepix's `256 x 256` integer pixel grid. It is
implemented in C, uses OpenMP when available, and is called through a Python
extension. It does not call scikit-learn, SciPy KDTree, or HDBSCAN for the
native result.

## Input and distance

The T3PA parser maps each hit to

```text
(x, y, time) = (MatrixIndex % 256,
                MatrixIndex // 256,
                ToA - FToA / 16)
```

with time in 25 ns ToA ticks. DBSCAN uses Euclidean distance in

```text
(x, y, time / time_scale)
```

`log1p(ToT)` and the raw timing fields are preserved in output but do not
affect neighborhood distance. Nonzero `Overflow` records are device markers,
not hits; the particle loader excludes them by default and records the skipped
count. The lower-level T3PA iterator remains lossless for diagnostics.

The implementation sorts hits by time, searches only valid local grid offsets,
marks core points, joins core components with union-find, and assigns border
points using DBSCAN semantics. Large files are processed in overlapping hit
windows and clusters touching the overlap are reconciled.

## Build

An editable install builds the extension:

```bash
python -m pip install -e .
```

This requires a C compiler and NumPy headers. Linux builds enable OpenMP by
default. Set `PARTICLE_DISABLE_OPENMP=1` for a serial build. If an existing
editable install lacks the extension, rebuild it explicitly:

```bash
python setup.py build_ext --inplace
```

The backend reports a clear error when the native extension is unavailable;
it does not silently substitute a different clustering algorithm.

## Canonical configuration

The active parameter file is
[`configs/teachers/dbscan_phase1_native_eps5_v001.json`](../configs/teachers/dbscan_phase1_native_eps5_v001.json):

| parameter | value | meaning |
|---|---:|---|
| `eps` | 5.0 | Radius in `(x, y, scaled time)`. |
| `min_samples` | 2 | A core hit needs itself and at least one neighbor. |
| `time_scale` | 0.625 | ToA ticks per scaled-time unit: 15.625 ns or 10 FToA subticks. |
| `full_scan_threshold` | 250,000 | Larger files use windowed processing. |
| `window_size` | 250,000 | Hits per large-file window. |
| `window_overlap` | 25,000 | Hits shared for cross-window reconciliation. |

Build one compressed NPZ shard per T3PA file:

```bash
particle-build-particles \
  --input local_data/raw \
  --index data/raw_data_index.csv \
  --params configs/teachers/dbscan_phase1_native_eps5_v001.json \
  --out local_data/processed/particles_aligned_time_eps5_v001 \
  --backend native-grid-dbscan \
  --threads 32 \
  --skip-existing
```

Use `--threads 0` to leave thread selection to the backend. Omit
`--skip-existing` when every shard must be rebuilt from the current code and
parameters.

## Output

The output directory contains `manifest.csv` and one `.particles.npz` file for
each input table. A shard preserves:

- per-hit `x`, `y`, relative fine time, raw ToA/ToT/FToA/overflow, source row,
  and particle ID (overflow is normally zero after marker filtering);
- particle-grouped hit offsets, particle IDs, hit counts, energy sums, and
  x/y/time bounds;
- labels in retained source-row order, with `-1` denoting noise; use the saved
  `hit_source_row` values when mapping back across filtered marker rows;
- source path, parameter JSON, timing formula/unit, and
  `source_special_records_skipped`.

Particle hits are grouped before noise rows in the stored hit arrays;
`particle_offsets` indexes only the non-noise groups. Use the saved offsets and
labels rather than assuming the source-table order remains unchanged.

## Limits and validation

DBSCAN labels are candidate partitions, not hand-verified physical particle
truth. Parameter changes can merge close events or split tracks. Data-loss
markers are filtered, but the missing intervals cannot be recovered. Keep the
canonical JSON with derived artifacts, review manifest warnings and skipped
marker counts, and validate representative crowded, long-span, and
multi-detector files before using shards as training targets.

Detailed evidence and historical comparisons are archived in the [final eps5
reports](../experimental_notes/dbscan/final_eps5/) and [older DBSCAN
investigations](../experimental_notes/dbscan/historical/).

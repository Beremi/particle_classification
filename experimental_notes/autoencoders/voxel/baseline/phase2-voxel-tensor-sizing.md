# Phase 2 Voxel Tensor Sizing

This is a sizing pass for replacing the ordered-hit/path autoencoder input with a dense 3D tensor over `(x, y, time)`.

Source data:

- particle shards: `local_data/processed/particles_aligned_time_eps5_v001/`
- separator: `native-grid-dbscan`
- teacher config: `configs/teachers/dbscan_phase1_native_eps5_v001.json`
- clustering time scale: `time_scale = 0.625` ToA ticks
- fine timestamp: `relative(ToA - FToA / 16)`, in 25 ns ToA ticks

The audit measured unique estimated particles directly from the NPZ shards, not duplicated Phase 2 training views.

## Dataset Extents

| item | value |
|---|---:|
| raw files | 711 |
| estimated particles | 3,464,792 |
| hits inside particles | 43,655,589 |
| median hits per particle | 5 |
| p95 hits per particle | 52 |
| p99 hits per particle | 111 |
| max hits per particle | 1,525 |

Overall axis-aligned particle extent quantiles:

| quantile | `x` span px | `y` span px | max XY span px | time span ticks |
|---:|---:|---:|---:|---:|
| 50% | 3 | 3 | 3 | 1.875 |
| 90% | 10 | 9 | 12 | 4.438 |
| 95% | 15 | 14 | 17 | 5.438 |
| 97.5% | 19 | 19 | 21 | 6.563 |
| 99% | 24 | 24 | 29 | 8.563 |
| 99.5% | 31 | 31 | 38 | 10.500 |
| 99.9% | 54 | 52 | 62 | 15.313 |
| max | 256 | 247 | 256 | 28.438 |

Large particles are rare but important. For particles with more than 512 hits, the median max-XY span is 65 px, the p95 max-XY span is 206.5 px, and the p99 max-XY span reaches the full 256 px detector width.

## Candidate Dense Tensor Coverage

Coverage uses a centered crop with 1 detector pixel per `x/y` bin. Time bins are shown in ToA ticks. `0.625` tick is the current DBSCAN-aligned time bin, equal to 10 FToA subticks or 15.625 ns.

| time bin | tensor | voxels | particle coverage | hit coverage | energy coverage |
|---:|---|---:|---:|---:|---:|
| 0.625 | `48 x 48 x 24` | 55,296 | 99.620% | 95.747% | 95.865% |
| 0.625 | `64 x 64 x 32` | 131,072 | 99.899% | 98.383% | 98.446% |
| 0.625 | `64 x 64 x 48` | 196,608 | 99.912% | 98.604% | 98.653% |
| 0.625 | `96 x 96 x 32` | 294,912 | 99.972% | 99.306% | 99.369% |
| 0.625 | `96 x 96 x 48` | 442,368 | 99.986% | 99.540% | 99.588% |
| 1.250 | `64 x 64 x 16` | 65,536 | 99.894% | 98.322% | 98.389% |
| 1.250 | `96 x 96 x 16` | 147,456 | 99.967% | 99.234% | 99.304% |
| 1.250 | `96 x 96 x 24` | 221,184 | 99.986% | 99.540% | 99.588% |

## Proposed First Tensor Size

Use this for the first dense-voxel autoencoder experiment:

```text
time x y x x = 32 x 64 x 64
time bin = 0.625 ToA ticks = 15.625 ns
x/y bin = 1 detector pixel
voxel value = summed normalized log1p(ToT) energy
```

Why this size:

- It covers 99.899% of estimated particles.
- It covers 98.383% of particle hits.
- It stays small enough for 3D tensor experiments: 131,072 scalar voxels.
- With float16 storage it is about 256 KiB per particle before activations.
- It matches the DBSCAN time scale, so the voxel grid is aligned with the current healthy separator.

The tail should not be ignored. About 3,516 particles fall outside `64 x 64 x 32`, and those contain about 705,984 hits. For them, use either:

- a second high-coverage route: `48 x 96 x 96`, or
- multiple overlapping centered crops/views per large particle, then pool their embeddings.

## Safer High-Coverage Size

If we want one dense tensor size with minimal cropping:

```text
time x y x x = 48 x 96 x 96
time bin = 0.625 ToA ticks
```

This covers 99.986% of particles and 99.540% of hits, but it has 442,368 scalar voxels. It is probably usable on the RTX 5090 with small batches, but it will make architecture iteration slower.

## Recommendation

Start with `32 x 64 x 64` and explicitly track overflow flags. Train/evaluate metrics separately for:

- fully contained particles,
- cropped particles,
- multi-view large particles.

If the dense tensor experiment looks promising, add a high-coverage `48 x 96 x 96` run only after the smaller model proves that voxelization improves reconstruction quality.

The compact audit artifacts are under:

- `local_data/processed/phase2_tensor_size_audit_v001/particle_extents.npz`
- `local_data/processed/phase2_tensor_size_audit_v001/candidate_tensor_coverage.csv`

# DBSCAN Aligned-Time eps5 Realignment

This note records the local correction after visual inspection of
`experimental_notes/dbscan/alternatives/native-grid-dbscan-aligned-time-crowded-file-08-7293-many-particles-100x-time-fixed-z-cubes.md`.

The inspected window 2 in that report looked like one continuous particle, but
the previous aligned-time candidate (`eps=2.5`, `time_scale=0.625`,
`min_samples=2`) split it into eight labels. The root cause was still local
DBSCAN strictness, not timestamp parsing: the fine timestamp is already
`ToA - FToA / 16`.

## Updated Candidate

Tracked config:
`configs/teachers/dbscan_aligned_time_candidate.json`

Parameters:

```json
{
  "eps": 5.0,
  "min_samples": 2,
  "time_scale": 0.625
}
```

The important change is `eps: 2.5 -> 5.0`. `time_scale=0.625` keeps the same
fine-time alignment as the previous candidate.

## Anchor Checks

File:
`08_thu_proton_daily_batch/toa_tot__r0000007293.t3pa`

| window | visual expectation | fine-voxel DBSCAN | aligned eps=2.5 | aligned eps=5.0 |
|---|---:|---:|---:|---:|
| fine bins `8350464000..8350489600` | 1 particle | 6 labels, 76.7% noise | 8 labels, 5.0% noise | 1 label, 0.0% noise |
| fine bins `3846784000..3846809600` | about 5 particles | 41 labels, 44.2% noise | 5 labels, 4.3% noise | 5 labels, 1.9% noise |

Full-file comparison on the same crowded file:

| setting | particles | noise fraction |
|---|---:|---:|
| fine-voxel DBSCAN candidate | 76,331 | 0.373 |
| aligned-time eps=2.5 | 13,632 | 0.027 |
| aligned-time eps=5.0 | 11,399 | 0.007 |

## Visual Outputs

- Corrected visual-one window:
  [native-grid-dbscan-aligned-time-eps5-crowded-file-08-7293-window2-cubes.md](native-grid-dbscan-aligned-time-eps5-crowded-file-08-7293-window2-cubes.md)
- Corrected broader crowded-file gallery:
  [native-grid-dbscan-aligned-time-eps5-crowded-file-08-7293-many-particles-100x-time-fixed-z-cubes.md](native-grid-dbscan-aligned-time-eps5-crowded-file-08-7293-many-particles-100x-time-fixed-z-cubes.md)

## Status

This is a better candidate, not a final frozen teacher yet. It should next be
audited over several files for both failures:

- remaining split failures: one visible trajectory divided into multiple labels;
- merge failures: separate visible trajectories combined into one label.

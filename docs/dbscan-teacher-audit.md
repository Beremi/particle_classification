# DBSCAN Teacher Quality Audit

This audit checks generated particle shards for separator-quality failures.
It is different from backend label-agreement validation: exact agreement only means two implementations produce the same labels, not that those labels are physically useful.

## Current Conclusion

`dbscan_v001` is not a trustworthy particle separator for Phase 2 training.
The native backend is still useful and exact, but the active parameter set
percolates through dense files and merges many independent events into one
component.

The clearest failures are:

- `data I05/sync__I05-W0044_r001.t3pa`: 6,275,049 hits become one particle.
- `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa`: 69,374 hits become 22 particles, with the largest particle containing 68,986 hits.
- `F08/tot_toa__r0000000000.t3pa`: the 10 largest particles shown in the Phase 2 demo are multi-clump structures, not single particles; 342 `dbscan_v001` particles in that file span more than 50,000,000 ToA ticks.

This means the Phase 2 embedding and autoencoder reports are useful only as
pipeline smoke tests until the separator is rebuilt from a corrected teacher.

## Why V001 Failed

The active parameters are:

```json
{
  "eps": 5.0,
  "min_samples": 3,
  "time_scale": 15000000.0
}
```

With these values, two same-pixel hits can be DBSCAN neighbors even when they
are separated by `eps * time_scale = 75,000,000` ToA ticks. DBSCAN connectivity
is transitive, so a dense file can form a chain of locally valid links that
turns an entire acquisition into one component.

The original tuning objective over-weighted parameter stability and noise
fraction. It did not include enough dense-file anti-percolation pressure, and
the tuning grid did not include small enough `time_scale` values.

## Candidate Retune Direction

A first candidate is tracked as
`configs/teachers/dbscan_v002_candidate.json`:

```json
{
  "eps": 1.25,
  "min_samples": 3,
  "time_scale": 50000.0
}
```

This is not frozen yet. It was selected from the re-audit because it removes
the obvious file-scale and long-time merges on the inspected small/medium
files, and on a 250,000-hit dense I05 sample it produced 18,710 particles with
4.64% noise and largest component 157 hits. Full-dataset validation is still
required, and the windowed DBSCAN wrapper needs to stay optimized for many
small components.

## Summary

- files audited: 711 / 711
- files with warnings: 706
- files with file-scale particles: 99
- total hits: 44,725,206
- total particles: 98,341
- max largest-particle fraction: 1.000000
- CSV: `local_data/diagnostics/dbscan_v001_quality_audit/particle_shard_quality.csv`

## Largest Offenders

| source | rows | particles | largest hits | largest fraction | long-span particles | warnings |
|---|---:|---:|---:|---:|---:|---|
| `data I05/sync__I05-W0044_r000.t3pa` | 5,515,566 | 1 | 5,515,566 | 1.000000 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `data I05/sync__I05-W0044_r001.t3pa` | 6,275,049 | 1 | 6,275,049 | 1.000000 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `data I05/sync__I05-W0044_r002.t3pa` | 2,156,669 | 1 | 2,156,669 | 1.000000 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `data M07/sync__M07-W0044_r001.t3pa` | 1,620,995 | 1 | 1,620,995 | 1.000000 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `data M07/sync__M07-W0044_r002.t3pa` | 551,514 | 1 | 551,514 | 1.000000 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `data M07/sync__M07-W0044_r000.t3pa` | 1,413,287 | 1 | 1,413,286 | 0.999999 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `13_tue_proton_daily_batch/toa_tot__r0000028552.t3pa` | 244,515 | 3 | 244,496 | 0.999922 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `08_thu_proton_daily_batch/toa_tot__r0000007293.t3pa` | 487,940 | 4 | 487,833 | 0.999781 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `13_tue_proton_daily_batch/toa_tot__r0000028555.t3pa` | 264,094 | 9 | 264,030 | 0.999758 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `08_thu_proton_daily_batch/toa_tot__r0000007296.t3pa` | 279,297 | 11 | 279,195 | 0.999635 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `13_tue_proton_daily_batch/toa_tot__r0000028554.t3pa` | 323,631 | 13 | 323,503 | 0.999604 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `D05/tot_toa__r0000000052.t3pa` | 1,430,116 | 74 | 1,429,457 | 0.999539 | 1 | `largest_fraction=1.000;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 62,851 | 4 | 62,817 | 0.999459 | 2 | `largest_fraction=0.999;file_scale_particles=1;long_span_particles=2;very_low_noise_fraction=0.000` |
| `08_thu_proton_daily_batch/toa_tot__r0000007295.t3pa` | 227,628 | 17 | 227,470 | 0.999306 | 1 | `largest_fraction=0.999;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `13_tue_proton_daily_batch/toa_tot__r0000028553.t3pa` | 208,020 | 15 | 207,864 | 0.999250 | 1 | `largest_fraction=0.999;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `D05/tot_toa__r0000000028.t3pa` | 1,334,602 | 126 | 1,333,291 | 0.999018 | 5 | `largest_fraction=0.999;file_scale_particles=1;long_span_particles=5;very_low_noise_fraction=0.000` |
| `08_thu_proton_daily_batch/toa_tot__r0000007332.t3pa` | 108,850 | 15 | 108,703 | 0.998650 | 1 | `largest_fraction=0.999;file_scale_particles=1;long_span_particles=1;very_low_noise_fraction=0.000` |
| `D05/tot_toa__r0000000038.t3pa` | 804,918 | 109 | 803,785 | 0.998592 | 4 | `largest_fraction=0.999;file_scale_particles=1;long_span_particles=4;very_low_noise_fraction=0.000` |
| `D05/tot_toa__r0000000027.t3pa` | 979,261 | 151 | 977,592 | 0.998296 | 4 | `largest_fraction=0.998;file_scale_particles=1;long_span_particles=4;very_low_noise_fraction=0.000` |
| `D05/tot_toa__r0000000046.t3pa` | 1,003,928 | 160 | 1,002,109 | 0.998188 | 7 | `largest_fraction=0.998;file_scale_particles=1;long_span_particles=7;very_low_noise_fraction=0.000` |

## Interpretation

Any row where one particle covers a large fraction of a file, spans the full detector, or has extremely long time extent is a DBSCAN percolation warning.
Those shards should not be used as trusted particle-level training data until the teacher parameters or separator are revised.

# Particle Separator Quality Audit

This audit checks generated particle shards for separator-quality failures.
It is different from backend label-agreement validation: exact agreement only means two implementations produce the same labels, not that those labels are physically useful.

## Summary

- files audited: 711 / 711
- files with warnings: 28
- files with file-scale particles: 0
- total hits: 44,725,206
- total particles: 4,036,167
- max largest-particle fraction: 0.684211
- CSV: `local_data/diagnostics/voxel_corner_min2_all_shards/particle_shard_quality.csv`

## Largest Offenders

| source | rows | particles | largest hits | largest fraction | long-span particles | warnings |
|---|---:|---:|---:|---:|---:|---|
| `08_thu_proton_daily_batch/toa_tot__r0000007334.t3pa` | 2,312 | 101 | 149 | 0.064446 | 0 | `largest_fraction=0.064` |
| `D05/tot_toa__r0000000004.t3pa` | 2,137 | 290 | 113 | 0.052878 | 0 | `largest_fraction=0.053` |
| `08_thu_proton_daily_batch/toa_tot__r0000006978.t3pa` | 35 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007007.t3pa` | 19 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007026.t3pa` | 19 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007030.t3pa` | 27 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007032.t3pa` | 21 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007042.t3pa` | 14 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007066.t3pa` | 17 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007085.t3pa` | 32 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007095.t3pa` | 16 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007111.t3pa` | 11 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007139.t3pa` | 14 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007155.t3pa` | 11 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007162.t3pa` | 20 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007190.t3pa` | 18 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007207.t3pa` | 8 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007209.t3pa` | 18 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007214.t3pa` | 17 | 0 | 0 | 0.000000 | 0 | `no_particles` |
| `08_thu_proton_daily_batch/toa_tot__r0000007235.t3pa` | 17 | 0 | 0 | 0.000000 | 0 | `no_particles` |

## Interpretation

Any statistically meaningful row where one particle covers a large fraction of a file, spans the full detector, or has extremely long time extent is a separator warning.
Tiny low-hit files can naturally have one continuous track covering much of the file; fraction warnings are therefore suppressed below the configured minimum row count.

# DBSCAN Continuity Audit

Continuity is measured in discrete `(x, y, time-bin)` space.
A particle label should be one connected component under the selected grid connectivity.
Different labels that touch under the same connectivity are reported as possible over-splits.

## Settings

- connectivity: `corner`
- time bin: `1.0` ToA ticks
- file CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/strict_eps1_ms2/continuity_audit/continuity_by_file.csv`
- disconnected-particle CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/strict_eps1_ms2/continuity_audit/disconnected_particles.csv`
- touching-label CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/strict_eps1_ms2/continuity_audit/touching_label_pairs.csv`

## Summary

- files audited: 5 / 5
- files with disconnected particles: 3
- total disconnected particles: 86
- max components inside one label: 5
- files with touching label pairs: 5
- total touching label pairs: 9293

## File Offenders

| source | particles | largest components | disconnected labels | touching label pairs | warnings |
|---|---:|---:|---:|---:|---|
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 6,888 | 3 | 71 | 3,610 | `disconnected_particles=71;touching_label_pairs=3610` |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 6,376 | 3 | 14 | 3,744 | `disconnected_particles=14;touching_label_pairs=3744` |
| `F08/tot_toa__r0000000047.t3pa` | 3,724 | 1 | 1 | 851 | `disconnected_particles=1;touching_label_pairs=851` |
| `F08/tot_toa__r0000000000.t3pa` | 4,093 | 1 | 0 | 936 | `touching_label_pairs=936` |
| `D05/tot_toa__r0000000005.t3pa` | 402 | 1 | 0 | 152 | `touching_label_pairs=152` |

## Largest Disconnected Labels

| source | particle | hits | components | time span |
|---|---:|---:|---:|---:|
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 2464 | 50 | 5 | 10687.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 203 | 9 | 4 | 1140.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 6551 | 17 | 4 | 33576.0 |
| `F08/tot_toa__r0000000047.t3pa` | 3612 | 4 | 3 | 22964.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 380 | 140 | 3 | 1166.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 3694 | 45 | 3 | 1032.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 383 | 3 | 3 | 1288.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 942 | 58 | 3 | 1244.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 1916 | 7 | 3 | 1872.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 2000 | 5 | 3 | 23744.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 2150 | 11 | 3 | 19242.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 2977 | 26 | 3 | 1164.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 3169 | 155 | 3 | 1258.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 3744 | 5 | 3 | 1186.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 3883 | 7 | 3 | 1480.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 6202 | 17 | 3 | 3967.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1405 | 30 | 2 | 1248.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1498 | 18 | 2 | 1144.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 2508 | 41 | 2 | 1125.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 2656 | 2 | 2 | 1315.0 |

## Strongest Touching Label Pairs

| source | particle A | particle B | touching voxel edges |
|---|---:|---:|---:|
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 2062 | 2064 | 92 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 4227 | 4229 | 78 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 4766 | 4768 | 76 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 4824 | 4827 | 76 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 4745 | 4746 | 74 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 6688 | 6690 | 74 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 497 | 499 | 72 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 5058 | 5061 | 70 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 6689 | 6690 | 70 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 1856 | 1858 | 68 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 3167 | 3169 | 68 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 3649 | 3651 | 68 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 5805 | 5807 | 68 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 379 | 380 | 66 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 4248 | 4249 | 66 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 3609 | 3610 | 64 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 5146 | 5147 | 64 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 6284 | 6286 | 64 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 4034 | 4035 | 62 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 397 | 398 | 62 |

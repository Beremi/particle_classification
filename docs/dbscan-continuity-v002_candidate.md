# DBSCAN Continuity Audit

Continuity is measured in discrete `(x, y, time-bin)` space.
A particle label should be one connected component under the selected grid connectivity.
Different labels that touch under the same connectivity are reported as possible over-splits.

## Settings

- connectivity: `corner`
- time bin: `1.0` ToA ticks
- file CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/v002_candidate/continuity_audit/continuity_by_file.csv`
- disconnected-particle CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/v002_candidate/continuity_audit/disconnected_particles.csv`
- touching-label CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/v002_candidate/continuity_audit/touching_label_pairs.csv`

## Summary

- files audited: 5 / 5
- files with disconnected particles: 5
- total disconnected particles: 4386
- max components inside one label: 27
- files with touching label pairs: 5
- total touching label pairs: 155

## File Offenders

| source | particles | largest components | disconnected labels | touching label pairs | warnings |
|---|---:|---:|---:|---:|---|
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 2,806 | 9 | 1,528 | 98 | `disconnected_particles=1528;touching_label_pairs=98` |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1,821 | 27 | 1,086 | 33 | `disconnected_particles=1086;touching_label_pairs=33` |
| `F08/tot_toa__r0000000000.t3pa` | 3,300 | 1 | 884 | 10 | `disconnected_particles=884;touching_label_pairs=10` |
| `F08/tot_toa__r0000000047.t3pa` | 2,932 | 1 | 784 | 9 | `disconnected_particles=784;touching_label_pairs=9` |
| `D05/tot_toa__r0000000005.t3pa` | 233 | 2 | 104 | 5 | `disconnected_particles=104;touching_label_pairs=5` |

## Largest Disconnected Labels

| source | particle | hits | components | time span |
|---|---:|---:|---:|---:|
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 108 | 291 | 27 | 1166.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1611 | 59 | 19 | 16387.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1440 | 47 | 16 | 16387.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1555 | 87 | 14 | 16385.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 806 | 140 | 14 | 18894.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1702 | 37 | 13 | 16386.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1773 | 58 | 13 | 16386.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 2607 | 34 | 13 | 16385.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 448 | 58 | 12 | 16384.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 605 | 44 | 12 | 16385.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1326 | 55 | 12 | 16387.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1627 | 59 | 12 | 16384.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 68 | 47 | 12 | 16385.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 404 | 53 | 11 | 16386.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 558 | 75 | 11 | 16389.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1764 | 44 | 11 | 16384.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 247 | 47 | 11 | 16387.0 |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 397 | 34 | 11 | 16384.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 444 | 45 | 10 | 16386.0 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 588 | 52 | 10 | 16386.0 |

## Strongest Touching Label Pairs

| source | particle A | particle B | touching voxel edges |
|---|---:|---:|---:|
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1223 | 1224 | 4 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1427 | 1428 | 4 |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 1684 | 1685 | 4 |
| `F08/tot_toa__r0000000000.t3pa` | 443 | 444 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 1024 | 1025 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 1070 | 1071 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 1245 | 1246 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 2103 | 2105 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 2346 | 2347 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 2695 | 2696 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 2813 | 2814 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 3001 | 3002 | 2 |
| `F08/tot_toa__r0000000000.t3pa` | 3071 | 3072 | 2 |
| `F08/tot_toa__r0000000047.t3pa` | 64 | 65 | 2 |
| `F08/tot_toa__r0000000047.t3pa` | 730 | 731 | 2 |
| `F08/tot_toa__r0000000047.t3pa` | 868 | 870 | 2 |
| `F08/tot_toa__r0000000047.t3pa` | 970 | 971 | 2 |
| `F08/tot_toa__r0000000047.t3pa` | 1716 | 1717 | 2 |
| `F08/tot_toa__r0000000047.t3pa` | 2007 | 2008 | 2 |
| `F08/tot_toa__r0000000047.t3pa` | 2011 | 2012 | 2 |

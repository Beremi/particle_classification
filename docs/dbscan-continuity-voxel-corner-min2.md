# DBSCAN Continuity Audit

Continuity is measured in discrete `(x, y, time-bin)` space.
A particle label should be one connected component under the selected grid connectivity.
Different labels that touch under the same connectivity are reported as possible over-splits.

## Settings

- connectivity: `corner`
- time bin: `1.0` ToA ticks
- file CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/voxel_corner_min2/continuity_audit/continuity_by_file.csv`
- disconnected-particle CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/voxel_corner_min2/continuity_audit/disconnected_particles.csv`
- touching-label CSV: `/home/beremi/repos/particle_classification/local_data/diagnostics/continuity_validation_sample/voxel_corner_min2/continuity_audit/touching_label_pairs.csv`

## Summary

- files audited: 5 / 5
- files with disconnected particles: 0
- total disconnected particles: 0
- max components inside one label: 1
- files with touching label pairs: 0
- total touching label pairs: 0

## File Offenders

| source | particles | largest components | disconnected labels | touching label pairs | warnings |
|---|---:|---:|---:|---:|---|
| `F08/tot_toa__r0000000000.t3pa` | 3,887 | 1 | 0 | 0 | `` |
| `F08/tot_toa__r0000000047.t3pa` | 3,515 | 1 | 0 | 0 | `` |
| `D05/tot_toa__r0000000005.t3pa` | 290 | 1 | 0 | 0 | `` |
| `13_tue_proton_daily_batch/toa_tot__r0000028689.t3pa` | 2,291 | 1 | 0 | 0 | `` |
| `08_thu_proton_daily_batch/toa_tot__r0000006981.t3pa` | 3,403 | 1 | 0 | 0 | `` |

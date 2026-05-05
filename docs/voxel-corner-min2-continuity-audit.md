# Particle Continuity Audit

Continuity is measured in discrete `(x, y, time-bin)` space.
A particle label should be one connected component under the selected grid connectivity.
Different labels that touch under the same connectivity are reported as possible over-splits.

## Settings

- connectivity: `corner`
- time bin: `1.0` ToA ticks
- file CSV: `local_data/diagnostics/voxel_corner_min2_all_continuity/continuity_by_file.csv`
- disconnected-particle CSV: `local_data/diagnostics/voxel_corner_min2_all_continuity/disconnected_particles.csv`
- touching-label CSV: `local_data/diagnostics/voxel_corner_min2_all_continuity/touching_label_pairs.csv`

## Summary

- files audited: 711 / 711
- files with disconnected particles: 0
- total disconnected particles: 0
- max components inside one label: 1
- files with touching label pairs: 0
- total touching label pairs: 0

## File Offenders

No continuity offenders were found.

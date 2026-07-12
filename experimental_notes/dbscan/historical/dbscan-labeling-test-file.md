# DBSCAN Labeling Visual Audit

Source file: `F08/tot_toa__r0000000000.t3pa`

This is the same file used in `phase2-autoencoder-reconstruction-demo.md`.
The plots show raw hits in 3D `(x, y, relative ToA)` with color representing the DBSCAN particle label.

## Summary

| label set | particles | noise hits | noise fraction | largest label hits | largest fraction | median hits | p95 hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dbscan_v001` | 1,872 | 419 | 1.93% | 124 | 0.57% | 7.0 | 32.0 |
| `dbscan_v002_candidate` | 3,300 | 1,811 | 8.36% | 62 | 0.29% | 5.0 | 12.0 |

## DBSCAN v001: All Labels

![DBSCAN v001 all labels](../../assets/dbscan_labeling_test_file/v001_all_dbscan_labels_3d.png)

## DBSCAN v001: Top 10 Largest Labels Highlighted

These are the labels that produced the suspicious large inputs in the autoencoder reconstruction page.

![DBSCAN v001 top 10 labels](../../assets/dbscan_labeling_test_file/v001_top10_largest_labels_3d.png)

## DBSCAN v001: Largest Label With Energy Color

![DBSCAN v001 largest particle energy](../../assets/dbscan_labeling_test_file/v001_largest_particle_energy_3d.png)

## Candidate Retuned DBSCAN: All Labels

![DBSCAN v002 candidate all labels](../../assets/dbscan_labeling_test_file/v002_candidate_all_dbscan_labels_3d.png)

## Side-By-Side Labeling

![DBSCAN v001 vs v002 candidate labels](../../assets/dbscan_labeling_test_file/v001_vs_v002_candidate_labels_3d.png)

## Interpretation

`dbscan_v001` is not healthy for this file: its largest labels are visibly multi-clump in 3D time space.
`dbscan_v002_candidate` is a diagnostic retune, not a frozen teacher yet, but it removes the most obvious long-time merged labels in this file.

## Correction: Split The Old Largest Label

The old `dbscan_v001` largest label is `p1365` with 124 hits. Under the retuned candidate it splits into 13 non-noise labels plus 8 noise hits, which matches the visual problem you pointed out much better than treating it as one particle.

| label set | full-file particles | noise fraction | largest full-file label | old p1365 split labels | old p1365 split noise |
|---|---:|---:|---:|---:|---:|
| `dbscan_v001` | 1,872 | 1.93% | 124 | 1 | 0 |
| `dbscan_v002_candidate` | 3,300 | 8.36% | 62 | 13 | 8 |
| strict `eps=1,min_samples=2` | 4,093 | 35.22% | 44 | 18 | 29 |

### Old p1365 Recolored By Retuned Candidate

![old p1365 recolored by v002](../../assets/dbscan_labeling_test_file/old_p1365_recolored_by_v002_candidate.png)

### Old p1365 Recolored By Strict DBSCAN

![old p1365 recolored by strict DBSCAN](../../assets/dbscan_labeling_test_file/old_p1365_recolored_by_strict_eps1_ms2.png)

### Retuned Candidate Top 10 Largest Labels

![v002 candidate top10](../../assets/dbscan_labeling_test_file/v002_candidate_top10_largest_labels_3d.png)

### Strict DBSCAN Top 10 Largest Labels

![strict top10](../../assets/dbscan_labeling_test_file/strict_eps1_ms2_top10_largest_labels_3d.png)

### Retuned Candidate Largest Current Label, Energy Colored

![v002 largest energy](../../assets/dbscan_labeling_test_file/v002_candidate_largest_particle_energy_3d.png)

### Strict DBSCAN All Labels

![strict all labels](../../assets/dbscan_labeling_test_file/strict_eps1_ms2_all_labels_3d.png)

### Strict DBSCAN Largest Current Label, Energy Colored

![strict largest energy](../../assets/dbscan_labeling_test_file/strict_eps1_ms2_largest_particle_energy_3d.png)

Interpretation: the retuned candidate fixes the old large multi-time merge for this file. The strict variant splits even more aggressively, but it also marks about 35% of hits as noise, so it is a visual/debug option rather than a frozen teacher.

## Continuity Criterion Result

Using the stated criterion directly, I labeled particles as connected components in discrete `(x, y, ToA)` voxels with 26-neighbor corner connectivity and `min_hits=2`. This is not DBSCAN density clustering; it is the exact continuity rule.

| label set | particles | noise fraction | largest label | disconnected labels under corner audit | touching label pairs |
|---|---:|---:|---:|---:|---:|
| `dbscan_v001` | 1,872 | 1.93% | 124 | many | not reliable |
| `dbscan_v002_candidate` | 3,300 | 8.36% | 62 | still present | low but not zero |
| `voxel_corner_min2` | 3,887 | 7.70% | 62 | 0 by construction | 0 by construction |

Old v001 largest `p1365` with 124 hits splits into 14 voxel-continuous labels plus 14 noise hits under `voxel_corner_min2`.

### Three-Way Comparison

![v001 vs v002 vs voxel corner comparison](../../assets/dbscan_labeling_test_file/v001_v002_voxel_corner_comparison_3d.png)

### Old p1365 Recolored By Voxel Corner Components

![old p1365 recolored by voxel corner components](../../assets/dbscan_labeling_test_file/old_p1365_recolored_by_voxel_corner_min2.png)

### Voxel Corner Components: All Labels

![voxel corner all labels](../../assets/dbscan_labeling_test_file/voxel_corner_min2_all_labels_3d.png)

### Voxel Corner Components: Top 10 Largest

![voxel corner top 10](../../assets/dbscan_labeling_test_file/voxel_corner_min2_top10_largest_labels_3d.png)

### Voxel Corner Largest Current Label, Energy Colored

![voxel corner largest energy](../../assets/dbscan_labeling_test_file/voxel_corner_min2_largest_particle_energy_3d.png)

Interpretation: if continuity in the time-space grid is the primary rule, the separator should be validated against this audit and probably use voxel connected components, possibly followed by domain-specific merging for known detector gaps. DBSCAN radius tuning alone cannot guarantee both no disconnected merges and no artificial cuts.

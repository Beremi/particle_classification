# Canonical Voxel AE Diagnostic

This diagnostic checks why the first canonical-transform voxel AE checkpoint looked good in scalar training metrics but bad in the reconstruction gallery.

## Short Answer

The checkpoint was selected on clean validation data, not on the noised training input. The problem is the validation loss itself: it used dense mean L1/MSE over the full `[32,64,64]` grid. Because the particles are extremely sparse, a reconstruction can miss most particle mass and still have a tiny dense-grid mean error. Total energy was learned well, but spatial energy placement was not.

The loss implementation has now been changed so future runs optimize energy-normalized mass error: `sum(abs(reconstruction-target)) / target_energy`, plus relative L2 and canonical mass L1. The old `v001` checkpoint should be treated as a failed/diagnostic run for shape reconstruction.

## Old Selection Metric

| metric from finished v001 run | value | interpretation |
|---|---:|---|
| best validation loss | 0.021829 | selected checkpoint step `128544` |
| old test dense final L1 | 0.00018772 | misleadingly tiny because it is averaged over 131,072 voxels |
| old test dense final MSE | 0.00049500 | also dominated by empty voxels |
| test energy relative error | 0.020266 | energy scale is good, but this does not prove shape reconstruction |

## Corrected Metrics On The Same Checkpoint

| corrected sampled test metric | value |
|---|---:|
| loss | 3.124611 |
| final_l1 | 1.792034 |
| final_l2_relative | 2.492400 |
| mass_overlap | 0.108001 |
| canonical_l1 | 1.969920 |
| energy_relative_l1 | 0.020987 |
| support_leakage | 0.009551 |
| final_mse | 0.000494 |

Under the corrected metric, `final_l1 ~= 1.79` is very bad. A perfect reconstruction is `0`; a mostly disjoint reconstruction with matching total energy approaches `2`. The mass overlap is only about `0.108`, so the visual failure is real.

## Gallery Example Diagnostics

| # | hits | energy err | old dense L1 | corrected rel L1 | rel L2 | mass overlap | target comps | recon comps | recon inside target support |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74 | 0.012 | 0.000435 | 1.708 | 2.452 | 0.152 | 1 | 1 | 1.000 |
| 2 | 50 | 0.031 | 0.000314 | 2.031 | 3.928 | 0.000 | 1 | 1 | 1.000 |
| 3 | 68 | 0.014 | 0.000484 | 1.843 | 3.243 | 0.085 | 7 | 1 | 1.000 |
| 4 | 56 | 0.019 | 0.000345 | 1.873 | 2.465 | 0.054 | 20 | 1 | 0.999 |
| 5 | 57 | 0.006 | 0.000425 | 1.864 | 4.483 | 0.065 | 1 | 1 | 1.000 |
| 6 | 136 | 0.002 | 0.000716 | 1.751 | 3.137 | 0.126 | 1 | 1 | 1.000 |
| 7 | 55 | 0.002 | 0.000337 | 1.676 | 2.532 | 0.161 | 1 | 1 | 1.000 |
| 8 | 85 | 0.009 | 0.000453 | 1.966 | 2.180 | 0.013 | 59 | 1 | 0.477 |
| 9 | 205 | 0.020 | 0.000978 | 1.763 | 2.975 | 0.128 | 1 | 1 | 1.000 |
| 10 | 161 | 0.035 | 0.001131 | 1.943 | 2.226 | 0.046 | 45 | 1 | 0.918 |
| 11 | 54 | 0.056 | 0.000375 | 1.925 | 1.872 | 0.066 | 7 | 1 | 0.989 |
| 12 | 71 | 0.014 | 0.000480 | 1.809 | 3.342 | 0.088 | 3 | 1 | 1.000 |

The table explains the contradiction: old dense L1 is around `0.0003-0.0011`, while corrected relative L1 is around `1.7-2.0`. The model often emits one compact component even when the target has multiple disconnected voxel components. Energy conservation alone hid that failure.

## Code Fix Applied

The training loss in `src/particle_classification/phase2_canonical_voxel_ae.py` now uses:

- final reconstruction relative L1: `sum(abs(recon-target)) / target_energy`;
- final relative L2: `sqrt(sum((recon-target)^2)) / sqrt(sum(target^2))`;
- canonical relative L1 over density mass, not dense voxel mean;
- mass overlap as a reported metric;
- the old dense MSE only as a secondary diagnostic.

A new training run is needed; the existing `v001` checkpoint should not be used as the Phase 2 shape encoder.

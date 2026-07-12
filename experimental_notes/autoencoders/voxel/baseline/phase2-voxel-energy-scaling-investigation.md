# Phase 2 Voxel Autoencoder Energy Scaling Investigation

The reconstruction gallery was not using separate color scales for the target and reconstruction rows. For each particle image, the clean target, corrupted input, raw reconstruction, and optional energy-matched reconstruction share one `magma` color scale. The absolute-error row uses its own error color scale.

So the visible energy mismatch is not just a plotting artifact. It is a real model behavior.

## Exact Gallery Energy Check

The table below measures the same ten held-out particles rendered in the energy diagnostic gallery.

| # | hits | clean E | input E | recon E | input/clean | recon/clean | recon relative error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74 | 33.400 | 32.763 | 58.026 | 0.981 | 1.737 | 0.737 |
| 2 | 50 | 20.271 | 19.956 | 30.791 | 0.984 | 1.519 | 0.519 |
| 3 | 68 | 34.430 | 34.646 | 61.001 | 1.006 | 1.772 | 0.772 |
| 4 | 56 | 24.129 | 24.513 | 28.561 | 1.016 | 1.184 | 0.184 |
| 5 | 57 | 29.908 | 29.304 | 41.900 | 0.980 | 1.401 | 0.401 |
| 6 | 136 | 53.553 | 53.447 | 109.869 | 0.998 | 2.052 | 1.052 |
| 7 | 55 | 26.329 | 26.389 | 43.100 | 1.002 | 1.637 | 0.637 |
| 8 | 85 | 30.218 | 30.428 | 27.509 | 1.007 | 0.910 | 0.090 |
| 9 | 205 | 72.712 | 72.489 | 142.054 | 0.997 | 1.954 | 0.954 |
| 10 | 161 | 76.315 | 76.186 | 139.879 | 0.998 | 1.833 | 0.833 |

The corrupted input energy is close to clean energy, so the corruption pipeline is not the main cause. The reconstruction often overestimates total energy, especially for particles above 50 hits.

## Bucketed Test-Sample Check

This check sampled up to 2,048 test particles per hit-count bucket from the same checkpoint and corruption setting.

| bucket | sampled | median recon/clean | mean recon/clean | mean relative error | p90 relative error |
|---|---:|---:|---:|---:|---:|
| 2-4 hits | 2,048 | 0.995 | 0.992 | 0.017 | 0.027 |
| 5-10 hits | 2,048 | 1.009 | 1.009 | 0.038 | 0.086 |
| 11-50 hits | 2,048 | 1.088 | 1.157 | 0.167 | 0.446 |
| 51+ hits | 2,048 | 1.650 | 1.712 | 0.714 | 1.190 |

This explains the apparent contradiction between the global test energy error and the gallery. The dataset is dominated by small particles, where energy scaling is good. The 50+ hit gallery intentionally selects larger particles, where the model has a large energy bias.

## Likely Cause

The model mostly captures geometry, but it spreads energy around the true support for larger particles. On the ten gallery examples, roughly one clean-energy equivalent is emitted outside the exact occupied target voxels, while the model still underfills the exact target voxels. That produces visually plausible shapes but poor total energy.

Likely contributing factors:

- Dense MSE is dominated by empty/background voxels and small particles.
- Larger particles are a minority of the training distribution.
- The softplus output is nonnegative everywhere and encourages low-level diffuse emission unless explicitly penalized.
- The current energy conservation term is too weak for large particles.
- The blur/noise denoising objective can reward smooth nearby energy instead of exact sparse energy placement.

## Diagnostic Gallery

The new gallery includes colorbars, per-particle energy sums, and a diagnostic energy-matched row:

[phase2-voxel-full-pass-energy-diagnostic-gallery.md](phase2-voxel-full-pass-energy-diagnostic-gallery.md)

The `recon E-matched` row is not the model output. It rescales the raw reconstruction to the corrupted-input energy, which helps separate shape quality from total-energy calibration.

## Recommended Fix

The next training pass should explicitly target energy calibration for medium and large particles:

- Increase sampling weight for `11-50` and `51+` hit particles.
- Increase the energy conservation loss, preferably bucket-weighted by hit count.
- Add a sparsity or support loss penalizing energy outside a dilated target support.
- Consider replacing softplus output with ReLU or a lower-background activation for this sparse voxel target.
- Report energy metrics by hit-count bucket, not only as one global average.

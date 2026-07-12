# Phase 2 Canonical Hard-Mined L2 Energy Diagnostic

The hard-mined high-corruption run looked good by its training metric, but the
noisy-input gallery showed reconstructions with far too much total energy.

The issue was not only plotting. The gallery table reads tensor sums directly:
several examples had clean/noisy energy around `20-120` but reconstructed energy
around `130-720`.

## Root Cause

The first hard-mined loss used an occupied/target-weighted voxel L2 with a very
small background weight (`0.02`). That made the model match the visible particle
path while hiding a large amount of low-amplitude energy over the empty
background. The old metric therefore reported a low value even when total
energy and outside-support energy were badly wrong.

On the regenerated noisy-input gallery batch:

| metric | old value |
|---|---:|
| old target-weighted L2 vs clean | `0.253` |
| plain relative L2 vs clean | `1.93` |
| mean total-energy relative error | `8.23x` |
| mean outside-support energy / target energy | `8.33x` |

## Correction

The hard-mined energy L2 loss was changed so future runs cannot hide energy in
the background:

- background voxel weight is now `1.0`, not `0.02`;
- total-energy relative error is included as an L2 scalar term;
- outside-support energy is included as an L2 scalar term;
- the model still trains only from final reconstruction terms, not canonical or
  transform supervision.

Applying the corrected loss to the same finished checkpoint and same gallery
batch gives:

| target | corrected loss | voxel L2 component | energy rel L1 | support leakage |
|---|---:|---:|---:|---:|
| clean target | `8.53` | `0.885` | `8.23` | `3.88` |
| noisy input | `7.31` | `4.27` | `5.82` | `0.59` |

So the checkpoint is not actually good in energy scaling; the earlier metric was
too forgiving. A corrected rerun should use the updated loss.

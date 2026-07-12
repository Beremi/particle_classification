# Phase 2 Canonical Transform Voxel Autoencoder

This is the replacement design for the unconstrained voxel autoencoder. The old model decoded from `concat(z_shape, z_aux)`, so rotation, scale, and energy could leak into the shape latent. The new model makes that harder by decoding only a canonical density from `z_shape`, then applying an explicit differentiable 3D transform.

## Input

Each particle is a centered `[T,Y,X] = [32,64,64]` tensor. The value in each voxel is summed normalized `log1p(ToT)` energy from the Phase 2 voxel cache:

```text
local_data/processed/phase2_voxel_energy_32x64x64_curriculum_v001
```

The tensor is treated as one scalar 3D field, not as a 4D feature tensor.

## Latent Contract

The model returns:

```text
z_shape:      8 values
z_transform: 7 values
```

`z_shape` is the only input to the canonical decoder. `z_transform` is never concatenated into the decoder hidden state.

The transform values are:

```text
dx_norm, dy_norm, dt_norm
theta_xy
theta_time_tilt
scale_xyz
energy_scale
```

`dx/dy/dt` are normalized-grid translations. `theta_xy` rotates in the detector plane. `theta_time_tilt` rotates in the canonical major-axis/time plane. `scale_xyz` is one linear uniform scale applied to `x,y,time`. `energy_scale` multiplies the final normalized density.

## Network

Encoder:

```text
[32,64,64] voxel tensor
  -> non-overlapping [4,8,8] patches
  -> patch MLP: 512 -> 192 -> 32
  -> flatten patch embeddings
  -> global MLP: 4096 -> 768 -> 768
  -> shape head: 768 -> 8
  -> transform head: 768 -> 7 constrained transform values
```

Decoder:

```text
z_shape only
  -> global decoder MLP: 8 -> 768 -> 4096
  -> patch decoder MLP: 32 -> 192 -> 512
  -> unpatchify to [32,64,64]
  -> softplus
  -> normalize to density sum 1
```

Fixed transform layer:

```text
canonical density
  -> affine_grid + grid_sample in x,y,time
  -> renormalize density sum to 1
  -> multiply by energy_scale
```

This is intentionally not a free decoder. The decoder can only make a centered canonical particle. Pose, scale, and total energy must pass through the explicit transform path.

## Supervision

For each clean target particle, weak transform targets are computed with weighted PCA:

- center gives `dx/dy/dt`;
- XY principal axis gives `theta_xy`;
- major-axis/time covariance gives `theta_time_tilt`;
- weighted RMS gives `scale_xyz`;
- total voxel energy gives `energy_scale`.

The clean particle is inverse-transformed with these PCA targets to create a canonical density target. This gives the model a direct loss on both:

- final transformed reconstruction vs the clean input;
- canonical decoded density vs the PCA-canonical target.

Synthetic transform augmentation is also used. A clean particle is randomly transformed with known translation, XY rotation, time tilt, scale, and energy. The augmented input is trained against its own PCA transform target, and its `z_shape` is pulled toward the original `z_shape`.

## Losses

The training objective combines:

- weighted final L1 and MSE reconstruction loss;
- canonical L1 loss;
- transform loss on translation, wrapped angles, scale, and energy;
- strong relative total-energy conservation loss;
- support leakage penalty outside a dilated target support;
- shape invariance between original and synthetically transformed views;
- augmented-view reconstruction/transform loss.

## Curriculum

Training starts from scratch. It uses 10 corruption levels from strong blur/noise to mild blur/noise:

```text
start: blur_kernel=5, blur_mix=1.0, noise_std=0.04, voxel_dropout=0.08
final: blur_mix=0.1, noise_std=0.004, voxel_dropout=0.008
```

At every level the optimizer is reset to AdamW with LR `1e-4`. Validation is checked every 512 steps. If validation plateaus for 4 evaluations, LR is halved. A level stops after LR decays below `5e-6`, with a minimum of 4096 and maximum of 20512 steps. Weights continue to the next corruption level, optimizer state does not.

## Commands

```bash
particle-train-canonical-voxel-ae \
  --cache local_data/processed/phase2_voxel_energy_32x64x64_curriculum_v001 \
  --out local_data/experiments/phase2_canonical_voxel_ae_v001 \
  --device cuda

particle-render-canonical-voxel-gallery \
  --checkpoint local_data/experiments/phase2_canonical_voxel_ae_v001/runs/canonical_transform_voxel_z8_t7/checkpoint.pt \
  --cache local_data/processed/phase2_voxel_energy_32x64x64_curriculum_v001 \
  --out experimental_notes/autoencoders/voxel/canonical/phase2-canonical-voxel-ae-gallery.md

particle-render-canonical-latent-pairs \
  --checkpoint local_data/experiments/phase2_canonical_voxel_ae_v001/runs/canonical_transform_voxel_z8_t7/checkpoint.pt \
  --cache local_data/processed/phase2_voxel_energy_32x64x64_curriculum_v001 \
  --out experimental_notes/autoencoders/voxel/canonical/phase2-canonical-latent-pair-gallery.md
```

## What To Inspect

The first acceptance check is not just reconstruction error. The important thing is whether pair galleries stop showing rotation stored in `z_shape`. Good pairs should have similar canonical reconstructions and visibly different explicit transform values.

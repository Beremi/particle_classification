# Phase 2 Voxel Autoencoder Experiment

This experiment tests whether the neural network has an easier reconstruction problem when each particle is represented as a centered 3D energy tensor instead of an ordered hit path.

## Input Tensor

Use one scalar value per voxel, so the logical input is a true 3D tensor:

```text
time x y x x = 32 x 64 x 64
```

Voxelization:

- `x/y`: 1 detector pixel per bin.
- `time`: `0.625` ToA ticks per bin, matching the current `native-grid-dbscan` separator.
- voxel value: summed normalized log energy, `log1p(ToT) / log1p(1023)`.
- centering: each particle is bbox-centered in `(x, y, time)` before voxelization.
- crop policy: hits outside the fixed cube are ignored.

Implementation note: PyTorch operations such as 3D blur temporarily add a singleton channel dimension internally, but the dataset and model interface treat each particle as `[T, Y, X]`.

The sizing audit showed that this grid covers about `99.899%` of particles and `98.383%` of hits. Every cache row stores `kept_fraction` and `kept_energy_fraction`, so cropped particles can be evaluated separately.

## Storage

The training cache is sparse on disk:

- `voxel_index`: flattened voxel index inside `32 x 64 x 64`.
- `voxel_value`: summed normalized log energy in that voxel.
- `voxel_offsets`: ragged offsets per particle.

Batches are densified into `[B, 32, 64, 64]` only when passed to the network.

## Model Family

Primary model: `patch_mlp`.

1. Split the cube into `4 x 8 x 8` patches.
2. Encode each patch with a shared MLP.
3. Flatten all patch embeddings.
4. Pass through a global MLP.
5. Keep the bottleneck structure:
   - `z_shape`: 8 dimensions
   - `z_aux`: 7 dimensions
6. Decode through a global MLP, then shared patch decoder.
7. Reassemble the `32 x 64 x 64` tensor.

This keeps the latent layout from the structured transform experiments while avoiding a huge single linear layer over all 131,072 voxels.

Baseline model: `flat_mlp`.

This flattens the entire cube directly. It is useful as a sanity baseline, but it has a much larger parameter count for the same hidden size.

## Multistage Training

The first training pass is a denoising curriculum with simple MSE against the clean tensor:

| stage | blur | noise | voxel dropout | purpose |
|---|---:|---:|---:|---|
| coarse | `5 x 5 x 5` | `0.04` | `0.08` | learn broad particle support |
| medium | `3 x 3 x 3` | `0.02` | `0.03` | sharpen shape and time structure |
| sharp | none | `0.004` | `0.00` | final clean reconstruction |

Loss:

```text
MSE(reconstruction, clean_energy_tensor)
```

The trainer also logs occupied-voxel MSE, background MSE, and total energy relative error. The default loss remains simple MSE, because this pass is meant to test whether voxel representation alone makes reconstruction easier.

## Commands

Build a bounded cache and run a smoke training pass:

```bash
python scripts/autoencoders/train_phase2_voxel_autoencoder.py \
  --particles local_data/processed/particles_aligned_time_eps5_v001 \
  --cache local_data/processed/phase2_voxel_energy_32x64x64_v001 \
  --out local_data/experiments/phase2_voxel_autoencoder_v001 \
  --rebuild-cache \
  --max-particles 50000 \
  --max-train-items 20000 \
  --max-val-items 5000 \
  --max-test-items 5000 \
  --batch-size 32 \
  --coarse-steps 500 \
  --medium-steps 750 \
  --sharp-steps 750 \
  --device cuda
```

Longer focused run:

```bash
python scripts/autoencoders/train_phase2_voxel_autoencoder.py \
  --particles local_data/processed/particles_aligned_time_eps5_v001 \
  --cache local_data/processed/phase2_voxel_energy_32x64x64_v001 \
  --out local_data/experiments/phase2_voxel_autoencoder_v001 \
  --max-train-items 250000 \
  --max-val-items 30000 \
  --max-test-items 30000 \
  --batch-size 64 \
  --hidden-dim 768 \
  --patch-hidden-dim 192 \
  --patch-embed-dim 32 \
  --coarse-steps 3000 \
  --medium-steps 5000 \
  --sharp-steps 5000 \
  --device cuda
```

Architecture sweep:

- `patch_mlp`, hidden `512`, patch embed `24`
- `patch_mlp`, hidden `768`, patch embed `32`
- `patch_mlp`, hidden `1024`, patch embed `48`
- `flat_mlp`, hidden `512` as a sanity baseline on a smaller subset

The first acceptance check is not physical classification yet. It is whether voxel reconstruction beats the path autoencoder on:

- total energy error,
- occupied-voxel MSE,
- large-particle reconstruction,
- qualitative cube plots of original vs reconstructed tensors.

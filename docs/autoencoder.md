# Particle autoencoders

Autoencoders operate on candidate particles produced by [Phase 1 DBSCAN](dbscan.md).
They compress morphology for reconstruction and exploratory grouping; they do
not turn unsupervised groups into physical particle-species labels.

Two approaches matter in the current repository:

| approach | role | representation | latent contract |
|---|---|---|---|
| Simple voxel z8 | Practical compact baseline | Centered `8 x 32 x 32` energy voxels | One unconstrained 8D code. |
| Pose-separated path | Research direction for XY invariance | Canonical ordered path with `(x, y, time, energy)` | Shape code excludes explicit detector-plane pose. |

## Practical baseline: simple voxel z8

The representative cache is derived from the larger `32 x 64 x 64` voxel
cache and compresses it to `8 x 32 x 32`: two detector pixels per x/y bin and
2.5 ToA ticks per time bin. Each particle is centered by its x/y/time bounding
box, and voxel values sum `log1p(ToT) / log1p(1023)`.

The model is a patch-MLP autoencoder with an eight-dimensional bottleneck and
no auxiliary or transform latent. The practical local checkpoint came from a
clean-data continuation with early best validation; later fixed-subset training
overfit. Its scripts are:

- [`build_compressed_voxel_subset.py`](../scripts/autoencoders/build_compressed_voxel_subset.py)
  for the representative cache;
- [`train_simple_voxel_gpu_cached_l2.py`](../scripts/autoencoders/train_simple_voxel_gpu_cached_l2.py)
  for the direct z8 model;
- [`render_phase2_simple_latent_groups.py`](../scripts/autoencoders/render_phase2_simple_latent_groups.py)
  for diagnostic latent grouping.

This is the easiest model to use for compact browsing and coarse morphology
grouping, but it has important limits:

- centering removes translation but does not guarantee rotation invariance;
- the fixed crop and downsampling lose detail, especially for large particles;
- z8 reconstruction is approximate and training is sensitive to corruption and
  hard-mined-subset overfitting;
- KMeans always creates groups, even on a continuous or nuisance-dominated
  manifold; group IDs are not particle types;
- checkpoints, caches, and latent arrays are local generated artifacts and are
  not shipped in Git.

See the archived [clean z8 training report](../experimental_notes/autoencoders/voxel/simple_z8/phase2-simple-voxel-z8-b1024-clean-fixedmine-20phase-training-convergence.md)
and [latent-group analysis](../experimental_notes/autoencoders/latent_analysis/phase2-simple-z8-latent-groups.md).

## Research approach: pose-separated paths

The pose-separated design makes the XY-invariance contract explicit:

1. resample a candidate as a time-ordered energy path;
2. estimate detector-plane orientation with energy-weighted PCA;
3. store `cos(theta_xy)`, `sin(theta_xy)`, and orientation reliability
   `q_theta` as metadata;
4. rotate the path into a canonical XY frame;
5. encode and decode the canonical path;
6. apply a fixed rotation to reconstruct the original local frame.

The learned shape latent can retain morphology, time profile, and energy
profile without using detector-plane angle as a class feature. Current code is
in the [path experiment module](../src/particle_classification/experiments/autoencoders/path.py)
with training entry point
[`train_phase2_pose_autoencoder.py`](../scripts/autoencoders/train_phase2_pose_autoencoder.py).

This structure is scientifically preferable when orientation must remain
separate, but it is not the practical compact baseline. Path ordering and PCA
orientation can be unstable for small or symmetric candidates, and high-fidelity
reconstruction required much larger latents than 8–16 dimensions. Some early
path reports also used coarse-time shards and are retained only as history.

See the current [pose-separated probe](../experimental_notes/autoencoders/path/phase2-pose-separated-autoencoder.md)
and [network structure note](../experimental_notes/autoencoders/path/phase2-nn-structure.md).

## Interpretation rules

- Keep source path, Phase 1 parameters, cache grid, split group, checkpoint
  configuration, and reconstruction metrics with every latent artifact.
- Split by source file/group to avoid near-duplicate leakage across train and
  evaluation sets.
- Compare neighbors in the original representation as well as latent space;
  latent proximity alone does not establish physical similarity.
- Never interpret DBSCAN candidates or unsupervised latent clusters as species
  truth without human labels or external experimental ground truth.
- Keep `theta_xy` outside any class distance intended to be XY invariant.

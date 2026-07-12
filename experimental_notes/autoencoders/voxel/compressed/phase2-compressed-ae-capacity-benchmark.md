# Phase 2 Compressed AE Capacity Enlargement Benchmark

This is a compute-only benchmark for larger `CanonicalTransformVoxelAE` variants on the compressed `8 x 32 x 32` voxel tensors. It uses real cached particles, batch size `4096`, AMP, AdamW, plain tensor L2, and forward+backward+optimizer steps on the RTX 5090.

The benchmark is meant to estimate training slowdown for bigger models, not reconstruction quality. Quality still needs a real training run.

## Results

| variant | params | z_shape | hidden | patch hidden | patch embed | ms/step | particles/s | slowdown | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_z8 | 2,300,335 | 8 | 768 | 192 | 32 | 23.06 | 177,651 | 1.00x | 3.31 |
| baseline_z16 | 2,312,631 | 16 | 768 | 192 | 32 | 23.07 | 177,537 | 1.00x | 3.31 |
| medium_z8 | 4,386,111 | 8 | 1024 | 256 | 48 | 24.27 | 168,787 | 1.05x | 3.56 |
| medium_z16 | 4,402,503 | 16 | 1024 | 256 | 48 | 24.31 | 168,475 | 1.05x | 3.56 |
| large_z16 | 8,974,935 | 16 | 1536 | 384 | 64 | 27.19 | 150,623 | 1.18x | 4.05 |
| xlarge_z16 | 17,242,487 | 16 | 2048 | 512 | 96 | 31.16 | 131,445 | 1.35x | 4.60 |

## Interpretation

- `baseline_z16` is basically free: same speed as the current `z_shape=8` model, with only about 12k extra parameters. If the 8D latent is the bottleneck, this is the cleanest first test.
- `medium_z16` is the best low-risk enlargement: about `1.9x` parameters, but only about `1.05x` step time.
- `large_z16` is the best serious enlargement: about `3.9x` parameters and about `1.18x` step time.
- `xlarge_z16` is still practical on this GPU, but it is probably bigger than we need for the next diagnostic run.

## Recommendation

Run two follow-up trainings:

1. `baseline_z16`: tests whether the true bottleneck is the 8D shape latent. Runtime should be essentially unchanged.
2. `large_z16`: tests whether decoder/encoder width is limiting reconstruction. Expected wall time is roughly `1.2x` baseline, so the previous 71-minute run would likely become about 85-95 minutes with the same schedule.

I would not jump directly to `xlarge_z16` until `large_z16` shows a visible gain in the gallery.

## Raw Benchmark Conditions

- Input: compressed representative cache, train split.
- Batch: `4096` particles.
- Tensor: `[8, 32, 32]`, fp16 under AMP.
- Loss: plain tensor L2 on final reconstruction.
- Measured: 12 optimizer steps after 3 warmup steps.
- Artifact: `local_data/experiments/phase2_canonical_voxel_ae_compressed_plain_l2_gpu_cached_v001/capacity_benchmark.json`.

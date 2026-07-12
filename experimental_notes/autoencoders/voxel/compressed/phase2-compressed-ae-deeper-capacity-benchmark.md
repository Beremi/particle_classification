# Phase 2 Deeper/Wider Compressed AE Speed Benchmark

This benchmark tests larger `CanonicalTransformVoxelAE` variants after adding configurable global encoder/decoder depth. It uses the compressed `8 x 32 x 32` particle tensors, batch size `512`, AMP, AdamW, and plain tensor L2 on the RTX 5090.

The current trained reference is `large_z16_ref`: `z_shape=16`, `hidden=1536`, `patch_hidden=384`, `patch_embed=64`, encoder hidden layers `1`, decoder hidden layers `0`.

## Results

| variant | params | z | hidden | patch hidden | patch embed | enc/dec depth | ms/step | particles/s | slowdown | est same schedule | peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large_z16_ref | 8,974,935 | 16 | 1536 | 384 | 64 | 1 / 0 | 3.53 | 144,960 | 1.00x | 14.1 min | 0.61 |
| large_z16_enc2_dec1 | 13,702,743 | 16 | 1536 | 384 | 64 | 2 / 1 | 3.99 | 128,255 | 1.13x | 15.9 min | 0.69 |
| large_z16_enc3_dec2 | 18,430,551 | 16 | 1536 | 384 | 64 | 3 / 2 | 4.81 | 106,528 | 1.36x | 19.2 min | 0.77 |
| wide_z24 | 17,275,263 | 24 | 2048 | 512 | 96 | 1 / 0 | 4.40 | 116,454 | 1.24x | 17.6 min | 0.77 |
| wide_z24_enc2_dec1 | 25,676,159 | 24 | 2048 | 512 | 96 | 2 / 1 | 5.62 | 91,041 | 1.59x | 22.5 min | 0.91 |
| huge_z32 | 35,447,719 | 32 | 3072 | 768 | 128 | 1 / 0 | 6.83 | 74,960 | 1.93x | 27.3 min | 1.13 |
| huge_z32_enc2_dec1 | 54,340,519 | 32 | 3072 | 768 | 128 | 2 / 1 | 8.25 | 62,072 | 2.34x | 33.0 min | 1.43 |

## Interpretation

- Adding one encoder hidden layer and one decoder hidden layer to the current `large_z16` costs only about `1.13x` per step. This is the safest depth test.
- The deeper `large_z16_enc3_dec2` is still practical at `1.36x`, but it may be unnecessary unless `enc2_dec1` helps.
- The wider `wide_z24` costs `1.24x`; it increases both latent capacity and MLP width without adding depth.
- The combined `wide_z24_enc2_dec1` costs `1.59x`; it is a strong candidate if we want one serious upgrade.
- The `huge_z32` variants fit, but `35M-54M` parameters are probably too large for the next diagnostic step unless smaller upgrades visibly saturate.

## Recommendation

Run `large_z16_enc2_dec1` next:

```text
shape_latent_dim = 16
hidden_dim = 1536
patch_hidden_dim = 384
patch_embed_dim = 64
encoder_hidden_layers = 2
decoder_hidden_layers = 1
batch_size = 512
```

Why this one: it adds real depth to both encoder and decoder, keeps the same latent size so the comparison is clean, and should only make the previous 14.1-minute schedule roughly 16 minutes. If that improves the gallery/L2, then try `wide_z24_enc2_dec1` as the next larger jump.

## Code Note

`CanonicalVoxelAEConfig` now has backward-compatible depth fields:

```text
encoder_hidden_layers: default 1
decoder_hidden_layers: default 0
```

Those defaults reproduce the existing architecture, so older checkpoints remain readable.

Raw benchmark artifact: `local_data/experiments/phase2_canonical_voxel_ae_compressed_plain_l2_large_z16_b512_v001/deeper_capacity_benchmark_b512.json`.

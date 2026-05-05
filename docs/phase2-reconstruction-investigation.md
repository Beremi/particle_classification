# Phase 2 Reconstruction Investigation

This note follows up on the failed visual reconstruction check from the largest corrected `voxel_corner_min2_v001` particles.

## Verdict

The quick Phase 2 autoencoder from the broad sweep was not good enough for faithful particle reconstruction. I added a stronger query decoder and ran focused reconstruction training, which improved the numbers and the plots, but the result is still not visually faithful enough to treat the decoder as a trustworthy shape reconstructor.

The current Phase 2 embeddings may still be useful for morphology grouping, but the autoencoder decoder should not be used as evidence that the latent space preserves full particle geometry.

## What Was Wrong

The previous best reconstruction checkpoint was:

- `26_deepsets_denoising_ae_z8`
- flat decoder
- 8-dimensional latent
- 64 decoded points
- 800 training steps

That was far too weak for the top corrected components. The inspected particles contain 1100-1500 hits and are sampled to 512 input hits. A 64-point flat decoder can get a tolerable Chamfer score while drawing a compact blurry prototype, because Chamfer does not strongly punish topology loss.

## Focused Training

I added a query decoder to `Phase2ParticleModel`. The old flat decoder maps `z -> one large output vector`. The new query decoder uses learned output queries and predicts each decoded point from `[z, query_i]`, which gives the decoder point-specific capacity while keeping the bottleneck explicit.

I also built a reconstruction-focused training manifest:

| Split | 1-3 | 4-10 | 11-50 | 51-512 | >512 sampled |
|---|---:|---:|---:|---:|---:|
| train | 6000 | 12000 | 30000 | 40000 | 764 |
| val | 1250 | 2500 | 6000 | 8000 | 220 |
| test | 1250 | 2500 | 6000 | 8000 | 420 |

This is intentionally harder than the broad sweep validation set and gives large particles enough presence during training.

## Runs

| Run | Latent | Decoder | Decoder points | Best step | Focus val loss |
|---|---:|---|---:|---:|---:|
| `26_deepsets_denoising_ae_z8` | 8 | flat | 64 | 800 | not comparable |
| `01_deepsets_denoising_ae_query_z8_p256` | 8 | query | 256 | 2500 | 0.9042 |
| `02_settransformer_denoising_ae_query_z8_p256` | 8 | query | 256 | 2300 | 0.8771 |
| `03_deepsets_denoising_ae_query_z16_p256` | 16 | query | 256 | 2500 | 0.8663 |
| `04_deepsets_ae_query_z16_p512` | 16 | query | 512 | 2300 | 0.7561 |
| `05_deepsets_ae_query_z64_p512_long` | 64 | query | 512 | 5600 | 0.7320 |

The best focused run is:

`local_data/experiments/phase2_reconstruction_focus_v001/runs/05_deepsets_ae_query_z64_p512_long/checkpoint.pt`

## Top-Largest Visual Check

The top-10 largest-particle Chamfer score improved, but the pictures still show that the decoder is too blurry and only partially recovers the time-layer structure.

| Model | Latent | Decoder points | Top-10 mean Chamfer | Top-10 median Chamfer | Top-10 max Chamfer |
|---|---:|---:|---:|---:|---:|
| old broad sweep | 8 | 64 | 0.4649 | 0.4508 | 0.6351 |
| focused query | 16 | 512 | 0.3693 | 0.3815 | 0.5385 |
| focused query long | 64 | 512 | 0.3089 | 0.2953 | 0.4892 |

| Rank | Hits | old z8/p64 | z16/p512 | z64/p512 |
|---:|---:|---:|---:|---:|
| 1 | 1523 | 0.5606 | 0.2787 | 0.1823 |
| 2 | 1482 | 0.4541 | 0.3411 | 0.4647 |
| 3 | 1386 | 0.4476 | 0.5385 | 0.4892 |
| 4 | 1344 | 0.3736 | 0.5165 | 0.3707 |
| 5 | 1233 | 0.4357 | 0.4220 | 0.3239 |
| 6 | 1219 | 0.6351 | 0.3728 | 0.2667 |
| 7 | 1196 | 0.4962 | 0.2531 | 0.1891 |
| 8 | 1195 | 0.3543 | 0.1888 | 0.1878 |
| 9 | 1134 | 0.4725 | 0.3903 | 0.4453 |
| 10 | 1114 | 0.4192 | 0.3916 | 0.1692 |

Latest visual gallery:

[phase2-reconstruction-focus-z64-v001.md](phase2-reconstruction-focus-z64-v001.md)

Previous focused z16 gallery:

[phase2-reconstruction-focus-v001.md](phase2-reconstruction-focus-v001.md)

## Interpretation

The z64 model is visibly better than the original z8/p64 model, especially on some particles where it recovers multiple time bands. But it still often compresses long thin bands into shorter, fuzzy, curved clouds. That means the decoder has not learned a faithful geometry-preserving representation.

The likely failure causes are:

- A single global latent has to encode many exact hit positions.
- Chamfer loss allows blurry average shapes and does not preserve discrete voxel continuity.
- The largest connected components are high-density, multi-layer structures; they are harder than ordinary compact particles.
- The current decoder is unconditional after `z`; it does not use an autoregressive sequence, occupancy grid, or topology constraint.
- The broad Phase 2 validation metric was too weak for this visual requirement.

## Recommendation

Do not use this autoencoder decoder as the Phase 2 quality proof.

For faithful reconstruction, the next model should change objective, not just train longer:

- use a voxel occupancy or sparse occupancy reconstruction target over local `(x, y, t)` bins;
- add topology/continuity losses, not just Chamfer;
- use sliced Wasserstein or Sinkhorn matching per time slice;
- consider a masked-hit reconstruction model that predicts missing hits from visible hits, rather than decoding the whole particle only from one latent;
- keep latent-size sweeps, but include 32/64/128 as reconstruction baselines before claiming that 4/8 dimensions are enough.

For unsupervised family discovery, the decoder can remain auxiliary, but the report should use prototype inspection and stability metrics as the main evidence.

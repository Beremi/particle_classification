# Simple z8 Latent Pair Histograms

Latent source: `local_data/experiments/phase2_simple_z8_latent_groups_v001/latents_and_groups_k32.npz`
Latent array: `z_shape`

This report shows every 2D pair projection of the 8D autoencoder latent space as a density histogram. The color scale is `log(1 + particle count)`, so both dense cores and tails remain visible.

## Summary

| item | value |
|---|---:|
| particles | 1,000,000 |
| latent dimensions | 8 |
| pair histograms | 28 |
| bins per axis | 180 |
| plot bounds | p0.2 to p99.8 per dimension |

## Overview

![all pair histograms](../../assets/phase2_simple_z8_latent_pair_histograms_v001/all_latent_pair_histograms.png)

## Correlations

![latent correlation matrix](../../assets/phase2_simple_z8_latent_pair_histograms_v001/latent_correlation_matrix.png)

## Pair Gallery

### z0 vs z1

- correlation: `+0.2683`
- nonempty histogram bins: `24,351`
- max bin count: `2,138`

![z0 z1 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z0_z1_hist2d.png)

### z0 vs z2

- correlation: `-0.2742`
- nonempty histogram bins: `23,989`
- max bin count: `5,146`

![z0 z2 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z0_z2_hist2d.png)

### z0 vs z3

- correlation: `-0.1747`
- nonempty histogram bins: `23,315`
- max bin count: `4,739`

![z0 z3 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z0_z3_hist2d.png)

### z0 vs z4

- correlation: `+0.0544`
- nonempty histogram bins: `24,227`
- max bin count: `4,883`

![z0 z4 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z0_z4_hist2d.png)

### z0 vs z5

- correlation: `+0.0808`
- nonempty histogram bins: `23,259`
- max bin count: `4,881`

![z0 z5 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z0_z5_hist2d.png)

### z0 vs z6

- correlation: `+0.4180`
- nonempty histogram bins: `22,009`
- max bin count: `4,054`

![z0 z6 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z0_z6_hist2d.png)

### z0 vs z7

- correlation: `+0.1285`
- nonempty histogram bins: `22,437`
- max bin count: `3,251`

![z0 z7 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z0_z7_hist2d.png)

### z1 vs z2

- correlation: `-0.1235`
- nonempty histogram bins: `27,131`
- max bin count: `2,553`

![z1 z2 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z1_z2_hist2d.png)

### z1 vs z3

- correlation: `-0.0050`
- nonempty histogram bins: `25,986`
- max bin count: `1,972`

![z1 z3 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z1_z3_hist2d.png)

### z1 vs z4

- correlation: `-0.2033`
- nonempty histogram bins: `25,013`
- max bin count: `3,479`

![z1 z4 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z1_z4_hist2d.png)

### z1 vs z5

- correlation: `+0.2212`
- nonempty histogram bins: `24,809`
- max bin count: `3,168`

![z1 z5 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z1_z5_hist2d.png)

### z1 vs z6

- correlation: `+0.2247`
- nonempty histogram bins: `25,122`
- max bin count: `2,045`

![z1 z6 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z1_z6_hist2d.png)

### z1 vs z7

- correlation: `-0.0396`
- nonempty histogram bins: `26,747`
- max bin count: `2,865`

![z1 z7 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z1_z7_hist2d.png)

### z2 vs z3

- correlation: `+0.3129`
- nonempty histogram bins: `23,310`
- max bin count: `4,878`

![z2 z3 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z2_z3_hist2d.png)

### z2 vs z4

- correlation: `-0.1099`
- nonempty histogram bins: `24,160`
- max bin count: `6,701`

![z2 z4 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z2_z4_hist2d.png)

### z2 vs z5

- correlation: `+0.0064`
- nonempty histogram bins: `23,607`
- max bin count: `5,975`

![z2 z5 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z2_z5_hist2d.png)

### z2 vs z6

- correlation: `-0.0917`
- nonempty histogram bins: `26,347`
- max bin count: `5,083`

![z2 z6 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z2_z6_hist2d.png)

### z2 vs z7

- correlation: `-0.0771`
- nonempty histogram bins: `22,422`
- max bin count: `4,466`

![z2 z7 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z2_z7_hist2d.png)

### z3 vs z4

- correlation: `-0.2037`
- nonempty histogram bins: `20,954`
- max bin count: `4,750`

![z3 z4 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z3_z4_hist2d.png)

### z3 vs z5

- correlation: `-0.0559`
- nonempty histogram bins: `22,789`
- max bin count: `7,012`

![z3 z5 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z3_z5_hist2d.png)

### z3 vs z6

- correlation: `-0.0984`
- nonempty histogram bins: `25,176`
- max bin count: `5,227`

![z3 z6 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z3_z6_hist2d.png)

### z3 vs z7

- correlation: `-0.0241`
- nonempty histogram bins: `21,229`
- max bin count: `3,667`

![z3 z7 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z3_z7_hist2d.png)

### z4 vs z5

- correlation: `-0.2525`
- nonempty histogram bins: `22,685`
- max bin count: `7,273`

![z4 z5 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z4_z5_hist2d.png)

### z4 vs z6

- correlation: `-0.3450`
- nonempty histogram bins: `23,193`
- max bin count: `4,724`

![z4 z6 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z4_z6_hist2d.png)

### z4 vs z7

- correlation: `+0.3178`
- nonempty histogram bins: `21,633`
- max bin count: `5,042`

![z4 z7 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z4_z7_hist2d.png)

### z5 vs z6

- correlation: `+0.0093`
- nonempty histogram bins: `24,721`
- max bin count: `6,372`

![z5 z6 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z5_z6_hist2d.png)

### z5 vs z7

- correlation: `-0.5484`
- nonempty histogram bins: `20,906`
- max bin count: `5,526`

![z5 z7 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z5_z7_hist2d.png)

### z6 vs z7

- correlation: `+0.2319`
- nonempty histogram bins: `24,386`
- max bin count: `3,566`

![z6 z7 histogram](../../assets/phase2_simple_z8_latent_pair_histograms_v001/pairs/z6_z7_hist2d.png)

## Notes

- These are model latent dimensions, not PCA axes.
- Bounds are percentile-clipped for visualization only; the full arrays remain saved in the local NPZ artifact.
- If you want clustering geometry, use `z_norm`; if you want the model's raw code values, use `z_shape`.
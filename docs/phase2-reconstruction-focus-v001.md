# Phase 2 Largest Particle Reconstruction Snapshots

This demo uses the corrected `voxel_corner_min2_v001` particle dataset and the selected Phase 2 reconstruction checkpoint.

- Dataset: `local_data/processed/phase2_particles_voxel_corner_min2_v001`
- Checkpoint: `local_data/experiments/phase2_reconstruction_focus_v001/runs/04_deepsets_ae_query_z16_p512/checkpoint.pt`
- Model: `deepsets` / `ae`
- Latent size: `16`
- Decoder output size: `512` points
- Numeric summary: [largest_reconstruction_summary.csv](assets/phase2_reconstruction_focus_v001/largest_reconstruction_summary.csv)

The left panel in each image is the actual Phase 2 input view, capped at 512 sampled hits for very large particles. The middle panel is the decoded 512-point point set. The right panel is the encoded latent vector `z`. Coordinates are particle-relative Phase 2 features: centered `x`, centered `y`, and scaled relative time. Color is normalized `log1p(ToT)` energy.

Important reading: the autoencoder is a compact point-set summarizer here. It is not expected to reproduce every hit of a 1000+ hit component; the useful question is whether the latent preserves enough morphology for downstream family grouping.

## Montage

![Top largest original decoded montage](assets/phase2_reconstruction_focus_v001/top_largest_original_decoded_montage.png)

## Top Largest Views

| Rank | Source | Particle | Hits | View hits | Sample | Chamfer | z preview | Image |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `F08/tot_toa__r0000000028.t3pa` | 8867 | 1523 | 512 | 0.336 | 0.2787 | `-0.064, -0.100, -0.280, -0.370` | [png](assets/phase2_reconstruction_focus_v001/particle_01_p008867_original_decoded.png) |
| 2 | `F08/tot_toa__r0000000028.t3pa` | 16244 | 1482 | 512 | 0.345 | 0.3411 | `0.108, -0.200, -0.030, -0.165` | [png](assets/phase2_reconstruction_focus_v001/particle_02_p016244_original_decoded.png) |
| 3 | `F08/tot_toa__r0000000052.t3pa` | 22538 | 1386 | 512 | 0.369 | 0.5385 | `0.157, -0.138, -0.038, -0.177` | [png](assets/phase2_reconstruction_focus_v001/particle_03_p022538_original_decoded.png) |
| 4 | `F08/tot_toa__r0000000028.t3pa` | 60858 | 1344 | 512 | 0.381 | 0.5165 | `-0.017, -0.176, -0.170, -0.148` | [png](assets/phase2_reconstruction_focus_v001/particle_04_p060858_original_decoded.png) |
| 5 | `F08/tot_toa__r0000000052.t3pa` | 95007 | 1233 | 512 | 0.415 | 0.4220 | `0.140, -0.172, -0.131, -0.180` | [png](assets/phase2_reconstruction_focus_v001/particle_05_p095007_original_decoded.png) |
| 6 | `F08/tot_toa__r0000000028.t3pa` | 5169 | 1219 | 512 | 0.420 | 0.3728 | `-0.018, -0.058, -0.031, -0.123` | [png](assets/phase2_reconstruction_focus_v001/particle_06_p005169_original_decoded.png) |
| 7 | `F08/tot_toa__r0000000052.t3pa` | 54455 | 1196 | 512 | 0.428 | 0.2531 | `-0.017, -0.153, 0.107, -0.213` | [png](assets/phase2_reconstruction_focus_v001/particle_07_p054455_original_decoded.png) |
| 8 | `F08/tot_toa__r0000000027.t3pa` | 18919 | 1195 | 512 | 0.428 | 0.1888 | `-0.304, -0.140, -0.369, -0.160` | [png](assets/phase2_reconstruction_focus_v001/particle_08_p018919_original_decoded.png) |
| 9 | `F08/tot_toa__r0000000052.t3pa` | 47929 | 1134 | 512 | 0.451 | 0.3903 | `0.155, -0.183, -0.042, -0.188` | [png](assets/phase2_reconstruction_focus_v001/particle_09_p047929_original_decoded.png) |
| 10 | `F08/tot_toa__r0000000028.t3pa` | 73903 | 1114 | 512 | 0.460 | 0.3916 | `-0.200, -0.169, -0.218, -0.169` | [png](assets/phase2_reconstruction_focus_v001/particle_10_p073903_original_decoded.png) |

### Rank 1: Particle 8867

![Particle 8867](assets/phase2_reconstruction_focus_v001/particle_01_p008867_original_decoded.png)

### Rank 2: Particle 16244

![Particle 16244](assets/phase2_reconstruction_focus_v001/particle_02_p016244_original_decoded.png)

### Rank 3: Particle 22538

![Particle 22538](assets/phase2_reconstruction_focus_v001/particle_03_p022538_original_decoded.png)

### Rank 4: Particle 60858

![Particle 60858](assets/phase2_reconstruction_focus_v001/particle_04_p060858_original_decoded.png)

### Rank 5: Particle 95007

![Particle 95007](assets/phase2_reconstruction_focus_v001/particle_05_p095007_original_decoded.png)

### Rank 6: Particle 5169

![Particle 5169](assets/phase2_reconstruction_focus_v001/particle_06_p005169_original_decoded.png)

### Rank 7: Particle 54455

![Particle 54455](assets/phase2_reconstruction_focus_v001/particle_07_p054455_original_decoded.png)

### Rank 8: Particle 18919

![Particle 18919](assets/phase2_reconstruction_focus_v001/particle_08_p018919_original_decoded.png)

### Rank 9: Particle 47929

![Particle 47929](assets/phase2_reconstruction_focus_v001/particle_09_p047929_original_decoded.png)

### Rank 10: Particle 73903

![Particle 73903](assets/phase2_reconstruction_focus_v001/particle_10_p073903_original_decoded.png)

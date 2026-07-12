# Phase 2 Particle Embedding Report

## Scope

Phase 2 starts after the active `voxel-cc-corner` separator has produced variable-size particles. The neural networks in this phase do not replace the separator; they learn embeddings for already separated `(x, y, time, energy)` hit sequences, then cluster those embeddings to discover particle families.

Discovered cluster IDs are morphology groups, not physical particle labels. They must be named later by inspection, simulation truth, or external labels.

## Current Run Status

The tracked metrics in this report come from the completed `fast` run artifacts in `local_data/experiments/phase2_particle_sweep_voxel_corner_min2_v001_representative`. Evaluation artifacts are in `local_data/experiments/phase2_particle_sweep_voxel_corner_min2_v001_representative/evaluation`.

Reproducible command sequence:

```bash
particle-train-phase2-sweep \
  --dataset local_data/processed/phase2_particles_voxel_corner_min2_v001 \
  --out local_data/experiments/phase2_particle_sweep_voxel_corner_min2_v001_representative \
  --budget fast \
  --latent-dim 4 --latent-dim 8 \
  --steps 800 \
  --max-train-items 200000 \
  --max-eval-items 2048 \
  --device cuda \
  --batch-size 96

particle-evaluate-phase2 \
  --dataset local_data/processed/phase2_particles_voxel_corner_min2_v001 \
  --experiment local_data/experiments/phase2_particle_sweep_voxel_corner_min2_v001_representative \
  --out local_data/experiments/phase2_particle_sweep_voxel_corner_min2_v001_representative/evaluation \
  --device cuda
```

## Data Flow

Raw `.t3pa` files are separated by `voxel_corner_min2_v001` / `voxel-cc-corner` into NPZ particle shards. `particle-build-phase2-dataset` converts every non-noise particle into one or more variable-hit views, with source-grouped train/validation/test splits by raw file. Large particles are sampled into multiple views capped at `max_points` while keeping full-particle summary descriptors.

- dataset views: 4037220
- particles represented: 4036167
- total view hits: 41546881
- split counts: `{'test': 563968, 'train': 3100807, 'val': 372445}`
- size buckets: `{'1-3': 1798826, '11-50': 701050, '4-10': 1380334, '51-512': 155606, '>512 sampled': 1404}`
- separator continuity audit: [`voxel-corner-min2-continuity-audit.md`](../../dbscan/historical/voxel-corner-min2-continuity-audit.md)
- separator shard audit: [`voxel-corner-min2-shard-audit.md`](../../dbscan/historical/voxel-corner-min2-shard-audit.md)

## Input Tensors

Per-hit point features: `x_centered`, `y_centered`, `t_scaled`, `t_norm`, `log_tot`, `ftoa_norm`, `r_norm`, `cos_theta`, `sin_theta`, `rank_time`.

Global summary features are concatenated after point pooling: `log_n_hits`, `log_total_energy`, `log_max_energy`, `mean_energy`, `std_energy`, `duration_scaled`, `log_duration`, `bbox_x`, `bbox_y`, `bbox_area`, `density_xy_time`, `pca_xy_linearity`, `pca_xy_minor`, `pca_xyt_linearity`, `pca_xyt_planarity`, `q_theta`, `cos_phi_t`, `sin_phi_t`, `sample_fraction`, `view_log_n_hits`, `large_flag`, `time_monotonicity`. Train-split mean/std statistics are stored in `normalization.json` and reused during training and evaluation.

## Model Family

The sweep supports four encoder backbones: `DeepSets/PointNet`, `EdgeConv/DGCNN-lite`, `SetTransformer-lite`, and `PointTransformer-lite`. Objectives cover point autoencoding, denoising autoencoding, masked point autoencoding, contrastive ParticleEmbed, DEC, and VaDE-style variational clustering. All models output a fixed-size latent vector `z` per particle view.

Training uses AdamW, cosine learning-rate scheduling, gradient clipping, automatic mixed precision on CUDA, hit dropout/jitter/time stretch augmentations where relevant, and stratified batches across hit-count buckets.

| Component | Implementation Detail |
|---|---|
| DeepSets/PointNet | shared hit MLP, max/mean/attention pooling, summary MLP, latent projection |
| EdgeConv/DGCNN-lite | shared hit MLP plus local kNN message block before pooling |
| SetTransformer-lite | two masked self-attention encoder layers over hit tokens |
| PointTransformer-lite | position-conditioned masked transformer over hit tokens |
| AE / denoising / masked AE | fixed-size point decoder trained with Chamfer-style reconstruction loss |
| Contrastive ParticleEmbed | two augmented views with InfoNCE alignment/uniformity objective |
| DEC | cluster-logit KL self-training target for frozen/fine-tuned embeddings |
| VaDE/GMM-VAE | variational latent with reconstruction, KL, and soft mixture entropy terms |

## Key Findings

- Internal sweep selection picked `11_edgeconv_dec_z4` (`edgeconv` + `dec`), with validation loss 0.0179 and selection score -0.0088.
- That internal selection is not a reconstruction-quality verdict: DEC/contrastive/VaDE losses are not numerically comparable to AE reconstruction losses.
- Best reconstruction-style encoder was `26_deepsets_denoising_ae_z8`, with validation loss 0.5122; this is the strongest candidate when preserving continuous particle geometry is the priority.
- Best HDBSCAN family-discovery row was `11_edgeconv_dec_z4` with 9 clusters, noise fraction 0.1909, silhouette 0.6577, and Davies-Bouldin 0.6393.
- Best fixed-K clustering row was `11_edgeconv_dec_z4` + `kmeans_12`, with silhouette 0.6169 and Davies-Bouldin 0.6054.
- Practical recommendation: inspect prototypes from the best DEC/HDBSCAN rows first for family discovery, and keep the best AE encoder as the geometry-preserving fallback representation. Do not assign physical class names until prototype inspection or external truth exists.

## Latent Compression

Reconstruction-style objectives provide the cleanest answer to the compression question because they directly penalize lost particle geometry. Lower validation loss is better; the values are comparable within this AE/denoising/masked-AE family.

| Latent dim | Best reconstruction run | Best val loss | Median recon val loss | Recon runs |
|---:|---|---:|---:|---:|
| 4 | `14_settransformer_denoising_ae_z4` | 0.6185 | 0.6433 | 12 |
| 8 | `26_deepsets_denoising_ae_z8` | 0.5122 | 0.5391 | 12 |

In this run, 8 latent dimensions preserve geometry substantially better than 4. A 4D latent is possible but visibly more compressed; use it only if the downstream clustering/prototype review says the loss of detail is acceptable.

## Validation Protocol

- training sample: 200000 source-grouped train views, sampled deterministically across the full manifest
- validation embeddings per run: 2048
- clustering metrics are unsupervised morphology checks; they do not prove physical particle identity
- separator health was checked before training: all 711 files had zero disconnected labels and zero touching-label pairs under the corner-continuity audit where exact touch checks were enabled

## Sweep Results

| Run | Backbone | Objective | Latent | Best Step | Val Loss | Embeddings | Selection |
|---|---|---|---:|---:|---:|---:|---:|
| `01_deepsets_ae_z4` | deepsets | ae | 4 | 800 | 0.6406 | 2048 | -0.6360 |
| `02_deepsets_denoising_ae_z4` | deepsets | denoising_ae | 4 | 800 | 0.6410 | 2048 | -0.6356 |
| `03_deepsets_masked_ae_z4` | deepsets | masked_ae | 4 | 800 | 0.6476 | 2048 | -0.6423 |
| `04_deepsets_contrastive_z4` | deepsets | contrastive | 4 | 200 | 1.9735 | 2048 | -1.9644 |
| `05_deepsets_dec_z4` | deepsets | dec | 4 | 600 | 0.0278 | 2048 | -0.0191 |
| `06_deepsets_vade_z4` | deepsets | vade | 4 | 800 | 0.9285 | 2048 | -0.9248 |
| `07_edgeconv_ae_z4` | edgeconv | ae | 4 | 800 | 0.6572 | 2048 | -0.6520 |
| `08_edgeconv_denoising_ae_z4` | edgeconv | denoising_ae | 4 | 800 | 0.6533 | 2048 | -0.6484 |
| `09_edgeconv_masked_ae_z4` | edgeconv | masked_ae | 4 | 800 | 0.6600 | 2048 | -0.6544 |
| `10_edgeconv_contrastive_z4` | edgeconv | contrastive | 4 | 200 | 1.9871 | 2048 | -1.9780 |
| `11_edgeconv_dec_z4` | edgeconv | dec | 4 | 800 | 0.0179 | 2048 | -0.0088 |
| `12_edgeconv_vade_z4` | edgeconv | vade | 4 | 800 | 0.9629 | 2048 | -0.9592 |
| `13_settransformer_ae_z4` | settransformer | ae | 4 | 800 | 0.6433 | 2048 | -0.6381 |
| `14_settransformer_denoising_ae_z4` | settransformer | denoising_ae | 4 | 800 | 0.6185 | 2048 | -0.6130 |
| `15_settransformer_masked_ae_z4` | settransformer | masked_ae | 4 | 800 | 0.6340 | 2048 | -0.6282 |
| `16_settransformer_contrastive_z4` | settransformer | contrastive | 4 | 600 | 1.9668 | 2048 | -1.9577 |
| `17_settransformer_dec_z4` | settransformer | dec | 4 | 800 | 0.0245 | 2048 | -0.0156 |
| `18_settransformer_vade_z4` | settransformer | vade | 4 | 800 | 0.9139 | 2048 | -0.9123 |
| `19_pointtransformer_ae_z4` | pointtransformer | ae | 4 | 700 | 0.6528 | 2048 | -0.6470 |
| `20_pointtransformer_denoising_ae_z4` | pointtransformer | denoising_ae | 4 | 800 | 0.6196 | 2048 | -0.6148 |
| `21_pointtransformer_masked_ae_z4` | pointtransformer | masked_ae | 4 | 800 | 0.6187 | 2048 | -0.6130 |
| `22_pointtransformer_contrastive_z4` | pointtransformer | contrastive | 4 | 700 | 2.0375 | 2048 | -2.0286 |
| `23_pointtransformer_dec_z4` | pointtransformer | dec | 4 | 800 | 0.0287 | 2048 | -0.0195 |
| `24_pointtransformer_vade_z4` | pointtransformer | vade | 4 | 800 | 0.8717 | 2048 | -0.8689 |
| `25_deepsets_ae_z8` | deepsets | ae | 8 | 800 | 0.5586 | 2048 | -0.5550 |
| `26_deepsets_denoising_ae_z8` | deepsets | denoising_ae | 8 | 800 | 0.5122 | 2048 | -0.5075 |
| `27_deepsets_masked_ae_z8` | deepsets | masked_ae | 8 | 800 | 0.5564 | 2048 | -0.5525 |
| `28_deepsets_contrastive_z8` | deepsets | contrastive | 8 | 700 | 1.2324 | 2048 | -1.2255 |
| `29_deepsets_dec_z8` | deepsets | dec | 8 | 800 | 0.0205 | 2048 | -0.0144 |
| `30_deepsets_vade_z8` | deepsets | vade | 8 | 800 | 0.7734 | 2048 | -0.7708 |
| `31_edgeconv_ae_z8` | edgeconv | ae | 8 | 800 | 0.5391 | 2048 | -0.5351 |
| `32_edgeconv_denoising_ae_z8` | edgeconv | denoising_ae | 8 | 800 | 0.5174 | 2048 | -0.5134 |
| `33_edgeconv_masked_ae_z8` | edgeconv | masked_ae | 8 | 800 | 0.5868 | 2048 | -0.5831 |
| `34_edgeconv_contrastive_z8` | edgeconv | contrastive | 8 | 500 | 1.2092 | 2048 | -1.2023 |
| `35_edgeconv_dec_z8` | edgeconv | dec | 8 | 700 | 0.0181 | 2048 | -0.0115 |
| `36_edgeconv_vade_z8` | edgeconv | vade | 8 | 800 | 0.7828 | 2048 | -0.7806 |
| `37_settransformer_ae_z8` | settransformer | ae | 8 | 800 | 0.5218 | 2048 | -0.5177 |
| `38_settransformer_denoising_ae_z8` | settransformer | denoising_ae | 8 | 800 | 0.5200 | 2048 | -0.5152 |
| `39_settransformer_masked_ae_z8` | settransformer | masked_ae | 8 | 800 | 0.5187 | 2048 | -0.5140 |
| `40_settransformer_contrastive_z8` | settransformer | contrastive | 8 | 300 | 1.2933 | 2048 | -1.2865 |
| `41_settransformer_dec_z8` | settransformer | dec | 8 | 800 | 0.0180 | 2048 | -0.0113 |
| `42_settransformer_vade_z8` | settransformer | vade | 8 | 800 | 0.8015 | 2048 | -0.7986 |
| `43_pointtransformer_ae_z8` | pointtransformer | ae | 8 | 800 | 0.5320 | 2048 | -0.5277 |
| `44_pointtransformer_denoising_ae_z8` | pointtransformer | denoising_ae | 8 | 800 | 0.5600 | 2048 | -0.5558 |
| `45_pointtransformer_masked_ae_z8` | pointtransformer | masked_ae | 8 | 800 | 0.5633 | 2048 | -0.5588 |
| `46_pointtransformer_contrastive_z8` | pointtransformer | contrastive | 8 | 800 | 1.2365 | 2048 | -1.2297 |
| `47_pointtransformer_dec_z8` | pointtransformer | dec | 8 | 700 | 0.0171 | 2048 | -0.0104 |
| `48_pointtransformer_vade_z8` | pointtransformer | vade | 8 | 800 | 0.7814 | 2048 | -0.7791 |

## Clustering Results

| Run | Latent | Method | Clusters | Noise | Silhouette | Davies-Bouldin |
|---|---:|---|---:|---:|---:|---:|
| `01_deepsets_ae_z4` | 4 | kmeans_8 | 8 | 0.000 | 0.352 | 0.900 |
| `01_deepsets_ae_z4` | 4 | gmm_8 | 8 | 0.000 | 0.261 | 1.110 |
| `01_deepsets_ae_z4` | 4 | kmeans_12 | 12 | 0.000 | 0.308 | 1.014 |
| `01_deepsets_ae_z4` | 4 | gmm_12 | 12 | 0.000 | 0.155 | 1.266 |
| `01_deepsets_ae_z4` | 4 | kmeans_16 | 16 | 0.000 | 0.300 | 1.009 |
| `01_deepsets_ae_z4` | 4 | gmm_16 | 16 | 0.000 | 0.170 | 1.391 |
| `01_deepsets_ae_z4` | 4 | kmeans_24 | 24 | 0.000 | 0.290 | 0.968 |
| `01_deepsets_ae_z4` | 4 | gmm_24 | 24 | 0.000 | 0.163 | 1.216 |
| `01_deepsets_ae_z4` | 4 | hdbscan | 4 | 0.110 | 0.375 | 0.614 |
| `02_deepsets_denoising_ae_z4` | 4 | kmeans_8 | 8 | 0.000 | 0.471 | 0.775 |
| `02_deepsets_denoising_ae_z4` | 4 | gmm_8 | 8 | 0.000 | 0.451 | 1.218 |
| `02_deepsets_denoising_ae_z4` | 4 | kmeans_12 | 12 | 0.000 | 0.462 | 0.704 |
| `02_deepsets_denoising_ae_z4` | 4 | gmm_12 | 12 | 0.000 | 0.410 | 0.985 |
| `02_deepsets_denoising_ae_z4` | 4 | kmeans_16 | 16 | 0.000 | 0.433 | 0.812 |
| `02_deepsets_denoising_ae_z4` | 4 | gmm_16 | 16 | 0.000 | 0.386 | 1.065 |
| `02_deepsets_denoising_ae_z4` | 4 | kmeans_24 | 24 | 0.000 | 0.401 | 0.755 |
| `02_deepsets_denoising_ae_z4` | 4 | gmm_24 | 24 | 0.000 | 0.370 | 1.017 |
| `02_deepsets_denoising_ae_z4` | 4 | hdbscan | 5 | 0.027 | 0.332 | 1.005 |
| `03_deepsets_masked_ae_z4` | 4 | kmeans_8 | 8 | 0.000 | 0.384 | 0.845 |
| `03_deepsets_masked_ae_z4` | 4 | gmm_8 | 8 | 0.000 | 0.396 | 0.938 |
| `03_deepsets_masked_ae_z4` | 4 | kmeans_12 | 12 | 0.000 | 0.363 | 0.922 |
| `03_deepsets_masked_ae_z4` | 4 | gmm_12 | 12 | 0.000 | 0.326 | 1.077 |
| `03_deepsets_masked_ae_z4` | 4 | kmeans_16 | 16 | 0.000 | 0.322 | 0.940 |
| `03_deepsets_masked_ae_z4` | 4 | gmm_16 | 16 | 0.000 | 0.312 | 1.117 |
| `03_deepsets_masked_ae_z4` | 4 | kmeans_24 | 24 | 0.000 | 0.291 | 0.988 |
| `03_deepsets_masked_ae_z4` | 4 | gmm_24 | 24 | 0.000 | 0.252 | 1.204 |
| `03_deepsets_masked_ae_z4` | 4 | hdbscan | 2 | 0.052 | 0.565 | 0.639 |
| `04_deepsets_contrastive_z4` | 4 | kmeans_8 | 8 | 0.000 | 0.274 | 1.121 |
| `04_deepsets_contrastive_z4` | 4 | gmm_8 | 8 | 0.000 | 0.167 | 1.219 |
| `04_deepsets_contrastive_z4` | 4 | kmeans_12 | 12 | 0.000 | 0.280 | 1.069 |
| `04_deepsets_contrastive_z4` | 4 | gmm_12 | 12 | 0.000 | 0.195 | 1.251 |
| `04_deepsets_contrastive_z4` | 4 | kmeans_16 | 16 | 0.000 | 0.279 | 1.042 |
| `04_deepsets_contrastive_z4` | 4 | gmm_16 | 16 | 0.000 | 0.173 | 1.196 |
| `04_deepsets_contrastive_z4` | 4 | kmeans_24 | 24 | 0.000 | 0.300 | 1.003 |
| `04_deepsets_contrastive_z4` | 4 | gmm_24 | 24 | 0.000 | 0.176 | 1.260 |
| `04_deepsets_contrastive_z4` | 4 | hdbscan | 6 | 0.448 | 0.214 | 0.980 |
| `05_deepsets_dec_z4` | 4 | kmeans_8 | 8 | 0.000 | 0.447 | 0.822 |
| `05_deepsets_dec_z4` | 4 | gmm_8 | 8 | 0.000 | 0.365 | 1.012 |
| `05_deepsets_dec_z4` | 4 | kmeans_12 | 12 | 0.000 | 0.449 | 0.819 |
| `05_deepsets_dec_z4` | 4 | gmm_12 | 12 | 0.000 | 0.399 | 0.998 |

## Plots

![latent reconstruction comparison](../../assets/phase2_particle_embedding/latent_reconstruction_comparison.png)

![01_deepsets_ae_z4 convergence](../../assets/phase2_particle_embedding/01_deepsets_ae_z4_convergence.png)

![02_deepsets_denoising_ae_z4 convergence](../../assets/phase2_particle_embedding/02_deepsets_denoising_ae_z4_convergence.png)

![03_deepsets_masked_ae_z4 convergence](../../assets/phase2_particle_embedding/03_deepsets_masked_ae_z4_convergence.png)

![04_deepsets_contrastive_z4 convergence](../../assets/phase2_particle_embedding/04_deepsets_contrastive_z4_convergence.png)

![05_deepsets_dec_z4 convergence](../../assets/phase2_particle_embedding/05_deepsets_dec_z4_convergence.png)

![06_deepsets_vade_z4 convergence](../../assets/phase2_particle_embedding/06_deepsets_vade_z4_convergence.png)

![07_edgeconv_ae_z4 convergence](../../assets/phase2_particle_embedding/07_edgeconv_ae_z4_convergence.png)

![08_edgeconv_denoising_ae_z4 convergence](../../assets/phase2_particle_embedding/08_edgeconv_denoising_ae_z4_convergence.png)

![09_edgeconv_masked_ae_z4 convergence](../../assets/phase2_particle_embedding/09_edgeconv_masked_ae_z4_convergence.png)

![10_edgeconv_contrastive_z4 convergence](../../assets/phase2_particle_embedding/10_edgeconv_contrastive_z4_convergence.png)

![11_edgeconv_dec_z4 convergence](../../assets/phase2_particle_embedding/11_edgeconv_dec_z4_convergence.png)

![12_edgeconv_vade_z4 convergence](../../assets/phase2_particle_embedding/12_edgeconv_vade_z4_convergence.png)

![13_settransformer_ae_z4 convergence](../../assets/phase2_particle_embedding/13_settransformer_ae_z4_convergence.png)

![14_settransformer_denoising_ae_z4 convergence](../../assets/phase2_particle_embedding/14_settransformer_denoising_ae_z4_convergence.png)

![15_settransformer_masked_ae_z4 convergence](../../assets/phase2_particle_embedding/15_settransformer_masked_ae_z4_convergence.png)

![16_settransformer_contrastive_z4 convergence](../../assets/phase2_particle_embedding/16_settransformer_contrastive_z4_convergence.png)

![17_settransformer_dec_z4 convergence](../../assets/phase2_particle_embedding/17_settransformer_dec_z4_convergence.png)

![18_settransformer_vade_z4 convergence](../../assets/phase2_particle_embedding/18_settransformer_vade_z4_convergence.png)

![19_pointtransformer_ae_z4 convergence](../../assets/phase2_particle_embedding/19_pointtransformer_ae_z4_convergence.png)

![20_pointtransformer_denoising_ae_z4 convergence](../../assets/phase2_particle_embedding/20_pointtransformer_denoising_ae_z4_convergence.png)

![21_pointtransformer_masked_ae_z4 convergence](../../assets/phase2_particle_embedding/21_pointtransformer_masked_ae_z4_convergence.png)

![22_pointtransformer_contrastive_z4 convergence](../../assets/phase2_particle_embedding/22_pointtransformer_contrastive_z4_convergence.png)

![23_pointtransformer_dec_z4 convergence](../../assets/phase2_particle_embedding/23_pointtransformer_dec_z4_convergence.png)

![24_pointtransformer_vade_z4 convergence](../../assets/phase2_particle_embedding/24_pointtransformer_vade_z4_convergence.png)

![25_deepsets_ae_z8 convergence](../../assets/phase2_particle_embedding/25_deepsets_ae_z8_convergence.png)

![26_deepsets_denoising_ae_z8 convergence](../../assets/phase2_particle_embedding/26_deepsets_denoising_ae_z8_convergence.png)

![27_deepsets_masked_ae_z8 convergence](../../assets/phase2_particle_embedding/27_deepsets_masked_ae_z8_convergence.png)

![28_deepsets_contrastive_z8 convergence](../../assets/phase2_particle_embedding/28_deepsets_contrastive_z8_convergence.png)

![29_deepsets_dec_z8 convergence](../../assets/phase2_particle_embedding/29_deepsets_dec_z8_convergence.png)

![30_deepsets_vade_z8 convergence](../../assets/phase2_particle_embedding/30_deepsets_vade_z8_convergence.png)

![31_edgeconv_ae_z8 convergence](../../assets/phase2_particle_embedding/31_edgeconv_ae_z8_convergence.png)

![32_edgeconv_denoising_ae_z8 convergence](../../assets/phase2_particle_embedding/32_edgeconv_denoising_ae_z8_convergence.png)

![33_edgeconv_masked_ae_z8 convergence](../../assets/phase2_particle_embedding/33_edgeconv_masked_ae_z8_convergence.png)

![34_edgeconv_contrastive_z8 convergence](../../assets/phase2_particle_embedding/34_edgeconv_contrastive_z8_convergence.png)

![35_edgeconv_dec_z8 convergence](../../assets/phase2_particle_embedding/35_edgeconv_dec_z8_convergence.png)

![36_edgeconv_vade_z8 convergence](../../assets/phase2_particle_embedding/36_edgeconv_vade_z8_convergence.png)

![37_settransformer_ae_z8 convergence](../../assets/phase2_particle_embedding/37_settransformer_ae_z8_convergence.png)

![38_settransformer_denoising_ae_z8 convergence](../../assets/phase2_particle_embedding/38_settransformer_denoising_ae_z8_convergence.png)

![39_settransformer_masked_ae_z8 convergence](../../assets/phase2_particle_embedding/39_settransformer_masked_ae_z8_convergence.png)

![40_settransformer_contrastive_z8 convergence](../../assets/phase2_particle_embedding/40_settransformer_contrastive_z8_convergence.png)

![41_settransformer_dec_z8 convergence](../../assets/phase2_particle_embedding/41_settransformer_dec_z8_convergence.png)

![42_settransformer_vade_z8 convergence](../../assets/phase2_particle_embedding/42_settransformer_vade_z8_convergence.png)

![43_pointtransformer_ae_z8 convergence](../../assets/phase2_particle_embedding/43_pointtransformer_ae_z8_convergence.png)

![44_pointtransformer_denoising_ae_z8 convergence](../../assets/phase2_particle_embedding/44_pointtransformer_denoising_ae_z8_convergence.png)

![45_pointtransformer_masked_ae_z8 convergence](../../assets/phase2_particle_embedding/45_pointtransformer_masked_ae_z8_convergence.png)

![46_pointtransformer_contrastive_z8 convergence](../../assets/phase2_particle_embedding/46_pointtransformer_contrastive_z8_convergence.png)

![47_pointtransformer_dec_z8 convergence](../../assets/phase2_particle_embedding/47_pointtransformer_dec_z8_convergence.png)

![48_pointtransformer_vade_z8 convergence](../../assets/phase2_particle_embedding/48_pointtransformer_vade_z8_convergence.png)

![cluster_quality](../../assets/phase2_particle_embedding/cluster_quality.png)

## Interpretation Checklist

- Prefer HDBSCAN for first-pass family discovery because it can leave ambiguous particles as noise.
- Compare discovered families by hit-count bucket, time span, total energy, PCA descriptors, and prototype galleries before assigning names.
- Train supervised imitator classifiers only after a discovered clustering is frozen; imitator accuracy is a deployment metric, not discovery truth.
- Treat file-scale or diffuse large clusters as quality-flagged candidates that may need separator review before physical interpretation.

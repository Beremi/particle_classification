# Simple z8 Voxel Autoencoder Gallery

Model input uses the same corruption parameters as training. This model has no explicit transform layer and only an 8D direct latent bottleneck.

Corruption: `blur_kernel=5`, `blur_mix=0.5`, `noise_std=0.02`, `voxel_dropout=0.04`.

| # | source | particle | hits | energy clean/noisy/recon | relative L2 clean | image |
|---:|---|---:|---:|---:|---:|---|
| 1 | `F08/tot_toa__r0000000033.t3pa` | 7583 | 431 | 128.751/135.646/123.986 | 0.2448 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_01.png) |
| 2 | `D05/tot_toa__r0000000039.t3pa` | 8867 | 86 | 36.759/43.348/31.922 | 0.3546 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_02.png) |
| 3 | `F08/tot_toa__r0000000033.t3pa` | 5075 | 53 | 24.913/29.403/22.239 | 0.3638 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_03.png) |
| 4 | `13_tue_proton_daily_batch/toa_tot__r0000028691.t3pa` | 1284 | 50 | 24.459/31.295/23.269 | 0.3259 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_04.png) |
| 5 | `08_thu_proton_daily_batch/toa_tot__r0000007293.t3pa` | 6857 | 97 | 48.082/58.521/48.899 | 0.4287 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_05.png) |
| 6 | `F08/tot_toa__r0000000033.t3pa` | 43762 | 51 | 23.601/26.145/21.147 | 0.3454 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_06.png) |
| 7 | `D05/tot_toa__r0000000045.t3pa` | 15295 | 59 | 27.508/32.384/24.678 | 0.3594 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_07.png) |
| 8 | `F08/tot_toa__r0000000033.t3pa` | 4402 | 57 | 25.182/30.191/22.778 | 0.2823 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_08.png) |
| 9 | `D05/tot_toa__r0000000039.t3pa` | 39380 | 80 | 36.463/44.713/37.121 | 0.3006 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_09.png) |
| 10 | `08_thu_proton_daily_batch/toa_tot__r0000007293.t3pa` | 9079 | 72 | 35.842/43.558/36.282 | 0.2198 | [png](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_10.png) |

## Example 1

![example 1](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_01.png)


## Example 2

![example 2](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_02.png)


## Example 3

![example 3](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_03.png)


## Example 4

![example 4](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_04.png)


## Example 5

![example 5](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_05.png)


## Example 6

![example 6](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_06.png)


## Example 7

![example 7](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_07.png)


## Example 8

![example 8](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_08.png)


## Example 9

![example 9](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_09.png)


## Example 10

![example 10](../../../assets/phase2_simple_voxel_z8_b16384_halfcorr_continue/gallery_v001/simple_z8_example_10.png)

# Phase 2 Real 50+ Hit Particle Autoencoder Gallery

These are held-out real Phase 2 particles from `local_data/processed/phase2_particles_eps5_v001`, filtered to `view_n_hits >= 50`. They are separator outputs from the final native-grid-DBSCAN Phase 1 baseline, not synthetic unit-test particles.

Each plot shows actual source hit points in the local centered/scaled particle frame. Color is normalized `log1p(ToT)`. The black curve is the 128-sample resampled target path used by the autoencoder. Red is the neural conv pose-separated autoencoder (`z=256`). Blue is the pose-separated PCA/basis autoencoder (`z=384`).

| rank | source | particle | hits | q_theta | neural err | PCA err | image |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `F08/tot_toa__r0000000028.t3pa` | 4365 | 512 | 0.999 | 0.4089 | 0.1022 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_01_p4365.png) |
| 2 | `F08/tot_toa__r0000000037.t3pa` | 12128 | 512 | 0.993 | 0.3842 | 0.0988 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_02_p12128.png) |
| 3 | `F08/tot_toa__r0000000028.t3pa` | 40181 | 512 | 0.999 | 0.3717 | 0.0918 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_03_p40181.png) |
| 4 | `F08/tot_toa__r0000000042.t3pa` | 29993 | 512 | 0.039 | 0.4828 | 0.0763 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_04_p29993.png) |
| 5 | `F08/tot_toa__r0000000037.t3pa` | 6283 | 512 | 0.012 | 0.4826 | 0.0960 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_05_p6283.png) |
| 6 | `F08/tot_toa__r0000000042.t3pa` | 25454 | 512 | 0.961 | 0.2888 | 0.0934 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_06_p25454.png) |
| 7 | `F08/tot_toa__r0000000028.t3pa` | 16903 | 74 | 0.460 | 0.2941 | 0.0173 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_07_p16903.png) |
| 8 | `F08/tot_toa__r0000000028.t3pa` | 37839 | 51 | 0.784 | 0.2539 | 0.0121 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_08_p37839.png) |
| 9 | `F08/tot_toa__r0000000028.t3pa` | 72151 | 60 | 0.837 | 0.2483 | 0.0107 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_09_p72151.png) |
| 10 | `F08/tot_toa__r0000000042.t3pa` | 12863 | 64 | 0.875 | 0.2446 | 0.0134 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_10_p12863.png) |
| 11 | `F08/tot_toa__r0000000028.t3pa` | 70393 | 125 | 0.869 | 0.3650 | 0.0894 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_11_p70393.png) |
| 12 | `F08/tot_toa__r0000000028.t3pa` | 63131 | 107 | 0.400 | 0.3123 | 0.0335 | [png](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_12_p63131.png) |

## Example 1

![example 1](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_01_p4365.png)

## Example 2

![example 2](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_02_p12128.png)

## Example 3

![example 3](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_03_p40181.png)

## Example 4

![example 4](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_04_p29993.png)

## Example 5

![example 5](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_05_p6283.png)

## Example 6

![example 6](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_06_p25454.png)

## Example 7

![example 7](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_07_p16903.png)

## Example 8

![example 8](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_08_p37839.png)

## Example 9

![example 9](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_09_p72151.png)

## Example 10

![example 10](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_10_p12863.png)

## Example 11

![example 11](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_11_p70393.png)

## Example 12

![example 12](../../assets/phase2_pose_path_ae_eps5/real_50hit_gallery/real_50hit_12_p63131.png)

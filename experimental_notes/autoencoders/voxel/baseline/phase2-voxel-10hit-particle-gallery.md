# Phase 2 Voxel Gallery: 10-Hit Particles

These are real estimated particles with exactly `10` hits, converted into logical 3D energy tensors of shape `32 x 64 x 64` (`time x y x x`).

Each image is a 2D view of the 3D tensor. The 32 time bins are split into four blocks of 8 bins, and each block is summed into one XY image.

Top row: clean centered energy tensor. Bottom row: the first curriculum-stage training input after `5 x 5 x 5` blur, support noise `0.04`, and voxel dropout `0.08`.

| # | source | particle | hits | kept hits | kept energy | image |
|---:|---|---:|---:|---:|---:|---|
| 1 | `F08/tot_toa__r0000000027.t3pa` | 9951 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_01.png) |
| 2 | `data I05/sync__I05-W0044_r002.t3pa` | 166544 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_02.png) |
| 3 | `data I05/sync__I05-W0044_r001.t3pa` | 196217 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_03.png) |
| 4 | `data I05/sync__I05-W0044_r000.t3pa` | 449138 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_04.png) |
| 5 | `data M07/sync__M07-W0044_r001.t3pa` | 329658 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_05.png) |
| 6 | `data I05/sync__I05-W0044_r000.t3pa` | 403309 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_06.png) |
| 7 | `data M07/sync__M07-W0044_r001.t3pa` | 273676 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_07.png) |
| 8 | `F08/tot_toa__r0000000048.t3pa` | 2852 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_08.png) |
| 9 | `data I05/sync__I05-W0044_r001.t3pa` | 412376 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_09.png) |
| 10 | `data I05/sync__I05-W0044_r000.t3pa` | 95255 | 10 | 100.00% | 100.00% | [png](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_10.png) |

## Example 1

![example 1](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_01.png)

## Example 2

![example 2](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_02.png)

## Example 3

![example 3](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_03.png)

## Example 4

![example 4](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_04.png)

## Example 5

![example 5](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_05.png)

## Example 6

![example 6](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_06.png)

## Example 7

![example 7](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_07.png)

## Example 8

![example 8](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_08.png)

## Example 9

![example 9](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_09.png)

## Example 10

![example 10](../../../assets/phase2_voxel_autoencoder/ten_hit_particle_gallery/voxel_training_example_10.png)

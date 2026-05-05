# Native Grid DBSCAN Fine-Time Voxel-Like Sample: Largest Labels

Fresh run after fixing T3PA timestamps to `time = ToA - FToA / 16`. Backend: `native-grid-dbscan`. Params: `eps=1.75`, `min_samples=2`, `time_scale=0.0625` ToA ticks (`1.5625 ns`). This is DBSCAN configured close to fine-time voxel continuity, not the old broad-time DBSCAN teacher.

Cube-grid plots with semitransparent occupied voxels and equal `(x, y, fine-time-bin)` scaling are in [native-grid-dbscan-fine-voxel-like-largest-cubes.md](native-grid-dbscan-fine-voxel-like-largest-cubes.md).

Output directory: `local_data/processed/native_grid_dbscan_fine_sample_v001/particles_fine_voxel_like`.

## Input Files

| source | hits | particles | noise frac | runtime s | hits/s | warnings |
|---|---:|---:|---:|---:|---:|---|
| `F08/tot_toa__r0000000028.t3pa` | 2,326,137 | 357,598 | 0.393 | 9.518 | 244,394 | `high_noise_fraction=0.393` |
| `F08/tot_toa__r0000000052.t3pa` | 2,452,962 | 378,140 | 0.394 | 10.091 | 243,080 | `high_noise_fraction=0.394` |
| `data M07/sync__M07-W0044_r000.t3pa` | 1,413,287 | 158,285 | 0.750 | 5.595 | 252,608 | `high_noise_fraction=0.750` |

![particle size histograms](assets/native_grid_dbscan_fine_voxel_like_sample_v001/particle_size_histograms.png)

## Largest Labels

Left = DBSCAN label colored by energy. Middle = same hits colored by strict face-connected components using one FToA bin (`1.5625 ns`). Right = time histogram.

| rank | source | particle | hits | span ns | face comps fine | corner comps fine | image |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `F08/tot_toa__r0000000028.t3pa` | 26944 | 403 | 21.88 | 296 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_01_tot_toa__r0000000028_p26944.png) |
| 2 | `F08/tot_toa__r0000000028.t3pa` | 20416 | 353 | 21.88 | 237 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_02_tot_toa__r0000000028_p20416.png) |
| 3 | `F08/tot_toa__r0000000052.t3pa` | 163637 | 298 | 17.19 | 206 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_03_tot_toa__r0000000052_p163637.png) |
| 4 | `F08/tot_toa__r0000000028.t3pa` | 12688 | 277 | 20.31 | 202 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_04_tot_toa__r0000000028_p12688.png) |
| 5 | `F08/tot_toa__r0000000028.t3pa` | 162344 | 270 | 18.75 | 195 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_05_tot_toa__r0000000028_p162344.png) |
| 6 | `F08/tot_toa__r0000000028.t3pa` | 148453 | 265 | 25.00 | 204 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_06_tot_toa__r0000000028_p148453.png) |
| 7 | `F08/tot_toa__r0000000028.t3pa` | 245297 | 263 | 18.75 | 178 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_07_tot_toa__r0000000028_p245297.png) |
| 8 | `F08/tot_toa__r0000000028.t3pa` | 261470 | 262 | 14.06 | 189 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_08_tot_toa__r0000000028_p261470.png) |
| 9 | `F08/tot_toa__r0000000028.t3pa` | 259800 | 258 | 20.31 | 188 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_09_tot_toa__r0000000028_p259800.png) |
| 10 | `F08/tot_toa__r0000000028.t3pa` | 160128 | 257 | 26.56 | 176 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_10_tot_toa__r0000000028_p160128.png) |
| 11 | `F08/tot_toa__r0000000052.t3pa` | 278678 | 256 | 20.31 | 176 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_11_tot_toa__r0000000052_p278678.png) |
| 12 | `F08/tot_toa__r0000000052.t3pa` | 183088 | 250 | 21.88 | 168 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_12_tot_toa__r0000000052_p183088.png) |
| 13 | `F08/tot_toa__r0000000052.t3pa` | 25847 | 248 | 15.62 | 166 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_13_tot_toa__r0000000052_p25847.png) |
| 14 | `F08/tot_toa__r0000000028.t3pa` | 113963 | 246 | 18.75 | 181 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_14_tot_toa__r0000000028_p113963.png) |
| 15 | `F08/tot_toa__r0000000052.t3pa` | 196028 | 233 | 17.19 | 148 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_15_tot_toa__r0000000052_p196028.png) |
| 16 | `F08/tot_toa__r0000000052.t3pa` | 234675 | 228 | 20.31 | 162 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_16_tot_toa__r0000000052_p234675.png) |
| 17 | `F08/tot_toa__r0000000052.t3pa` | 56021 | 226 | 14.06 | 152 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_17_tot_toa__r0000000052_p56021.png) |
| 18 | `F08/tot_toa__r0000000052.t3pa` | 165086 | 217 | 21.88 | 144 | 1 | [png](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_18_tot_toa__r0000000052_p165086.png) |

### Rank 1

![rank 1](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_01_tot_toa__r0000000028_p26944.png)

### Rank 2

![rank 2](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_02_tot_toa__r0000000028_p20416.png)

### Rank 3

![rank 3](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_03_tot_toa__r0000000052_p163637.png)

### Rank 4

![rank 4](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_04_tot_toa__r0000000028_p12688.png)

### Rank 5

![rank 5](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_05_tot_toa__r0000000028_p162344.png)

### Rank 6

![rank 6](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_06_tot_toa__r0000000028_p148453.png)

### Rank 7

![rank 7](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_07_tot_toa__r0000000028_p245297.png)

### Rank 8

![rank 8](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_08_tot_toa__r0000000028_p261470.png)

### Rank 9

![rank 9](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_09_tot_toa__r0000000028_p259800.png)

### Rank 10

![rank 10](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_10_tot_toa__r0000000028_p160128.png)

### Rank 11

![rank 11](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_11_tot_toa__r0000000052_p278678.png)

### Rank 12

![rank 12](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_12_tot_toa__r0000000052_p183088.png)

### Rank 13

![rank 13](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_13_tot_toa__r0000000052_p25847.png)

### Rank 14

![rank 14](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_14_tot_toa__r0000000028_p113963.png)

### Rank 15

![rank 15](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_15_tot_toa__r0000000052_p196028.png)

### Rank 16

![rank 16](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_16_tot_toa__r0000000052_p234675.png)

### Rank 17

![rank 17](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_17_tot_toa__r0000000052_p56021.png)

### Rank 18

![rank 18](assets/native_grid_dbscan_fine_voxel_like_sample_v001/largest_18_tot_toa__r0000000052_p165086.png)

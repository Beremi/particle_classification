# Reasonable File Fixed-Detector Fine-Time Cube Overview: 08 Thu 7333

Source shard: `local_data/processed/native_grid_dbscan_reasonable_file_v001/particles_fine_voxel_like/08_thu_proton_daily_batch/toa_tot__r0000007333.particles.npz`

Raw source: `local_data/raw/08_thu_proton_daily_batch/toa_tot__r0000007333.t3pa`

This is a whole-file inspection view. A literal one-image, equal-axis rendering of every cube is not useful because this file spans `12,777,351,155` fine-time bins. The report therefore uses whole-file summaries plus an equal-cube atlas of the densest short time windows.

- hits: `88,825`
- particles: `15,160`
- noise fraction: `0.434`
- fine-time span: `12,777,351,155` bins = `19.965` s

## Whole-File Context

![time density](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_time_density.png)

![xy density](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_xy_density.png)

![particle size over time](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_particle_time_size.png)

## Densest Time Windows As Equal Cubes

Each window uses real cube scaling: one x pixel, one y pixel, and one fine time bin have the same drawn size. The z axis is local to the selected time window, but the table gives absolute relative acquisition time.

The x/y axes are fixed to the full detector plane `0..256`.

| rank | start s | span ns | hits | voxels | particles | noise frac | image |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16.367674 | 400.0 | 610 | 610 | 63 | 0.243 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_01_z10475311104_10475311360_cubes.png) |
| 2 | 11.350905 | 400.0 | 307 | 307 | 28 | 0.248 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_02_z7264579328_7264579584_cubes.png) |
| 3 | 12.671127 | 400.0 | 210 | 210 | 32 | 0.324 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_03_z8109521408_8109521664_cubes.png) |
| 4 | 15.979692 | 400.0 | 183 | 183 | 16 | 0.322 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_04_z10227003136_10227003392_cubes.png) |
| 5 | 17.598084 | 400.0 | 151 | 151 | 18 | 0.265 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_05_z11262773760_11262774016_cubes.png) |
| 6 | 19.519795 | 400.0 | 149 | 149 | 23 | 0.430 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_06_z12492668672_12492668928_cubes.png) |
| 7 | 12.912553 | 400.0 | 147 | 147 | 18 | 0.143 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_07_z8264033792_8264034048_cubes.png) |
| 8 | 18.854454 | 400.0 | 145 | 145 | 18 | 0.331 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_08_z12066850560_12066850816_cubes.png) |
| 9 | 13.653522 | 400.0 | 140 | 140 | 20 | 0.300 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_09_z8738254080_8738254336_cubes.png) |
| 10 | 13.151577 | 400.0 | 139 | 139 | 18 | 0.237 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_10_z8417009152_8417009408_cubes.png) |
| 11 | 17.503425 | 400.0 | 134 | 134 | 18 | 0.269 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_11_z11202191872_11202192128_cubes.png) |
| 12 | 19.166740 | 400.0 | 133 | 133 | 15 | 0.195 | [png](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_12_z12266713856_12266714112_cubes.png) |

### Window 1

![window 1](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_01_z10475311104_10475311360_cubes.png)

### Window 2

![window 2](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_02_z7264579328_7264579584_cubes.png)

### Window 3

![window 3](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_03_z8109521408_8109521664_cubes.png)

### Window 4

![window 4](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_04_z10227003136_10227003392_cubes.png)

### Window 5

![window 5](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_05_z11262773760_11262774016_cubes.png)

### Window 6

![window 6](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_06_z12492668672_12492668928_cubes.png)

### Window 7

![window 7](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_07_z8264033792_8264034048_cubes.png)

### Window 8

![window 8](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_08_z12066850560_12066850816_cubes.png)

### Window 9

![window 9](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_09_z8738254080_8738254336_cubes.png)

### Window 10

![window 10](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_10_z8417009152_8417009408_cubes.png)

### Window 11

![window 11](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_11_z11202191872_11202192128_cubes.png)

### Window 12

![window 12](../../assets/native_grid_dbscan_fine_voxel_like_reasonable_08_7333/whole_file_window_12_z12266713856_12266714112_cubes.png)

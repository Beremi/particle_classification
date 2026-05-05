# Crowded File Many-Particle Fixed 256x256xTime Cube Overview: 08 Thu 7293, 100x Time

Source shard: `local_data/processed/native_grid_dbscan_crowded_file_v001/particles_fine_voxel_like/08_thu_proton_daily_batch/toa_tot__r0000007293.particles.npz`

Raw source: `local_data/raw/08_thu_proton_daily_batch/toa_tot__r0000007293.t3pa`

This is a whole-file inspection view. A literal one-image, equal-axis rendering of every cube is not useful because this file spans `12,327,013,936` fine-time bins. The report therefore uses whole-file summaries plus an equal-cube atlas of the densest short time windows.

- hits: `487,940`
- particles: `76,331`
- noise fraction: `0.373`
- fine-time span: `12,327,013,936` bins = `19.261` s
- plotted z cube: `100` fine bins = `156.2500` ns

## Whole-File Context

![time density](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_time_density.png)

![xy density](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_xy_density.png)

![particle size over time](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_particle_time_size.png)

## Densest Time Windows As Equal Cubes

Each window uses real cube scaling: one x pixel, one y pixel, and one fine time bin have the same drawn size. The z axis is local to the selected time window, but the table gives absolute relative acquisition time.

The x/y axes are fixed to the full detector plane `0..256`.
The z axis is fixed to the full selected time slab.

Windows were selected by `particles`.

| rank | start s | span ns | z cube ns | hits | voxels | particles | noise frac | image |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 10.960360 | 40000.0 | 156.2500 | 955 | 955 | 120 | 0.371 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_01_z7014630400_7014656000_cubes.png) |
| 2 | 7.944360 | 40000.0 | 156.2500 | 313 | 313 | 49 | 0.396 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_02_z5084390400_5084416000_cubes.png) |
| 3 | 5.994720 | 40000.0 | 156.2500 | 277 | 277 | 47 | 0.347 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_03_z3836620800_3836646400_cubes.png) |
| 4 | 16.665040 | 40000.0 | 156.2500 | 394 | 394 | 43 | 0.365 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_04_z10665625600_10665651200_cubes.png) |
| 5 | 6.510120 | 40000.0 | 156.2500 | 291 | 291 | 43 | 0.364 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_05_z4166476800_4166502400_cubes.png) |
| 6 | 6.527040 | 40000.0 | 156.2500 | 375 | 375 | 42 | 0.216 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_06_z4177305600_4177331200_cubes.png) |
| 7 | 6.930320 | 40000.0 | 156.2500 | 367 | 367 | 42 | 0.264 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_07_z4435404800_4435430400_cubes.png) |
| 8 | 8.515040 | 40000.0 | 156.2500 | 289 | 289 | 41 | 0.408 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_08_z5449625600_5449651200_cubes.png) |
| 9 | 6.025520 | 40000.0 | 156.2500 | 277 | 277 | 41 | 0.412 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_09_z3856332800_3856358400_cubes.png) |
| 10 | 8.253360 | 40000.0 | 156.2500 | 236 | 236 | 41 | 0.335 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_10_z5282150400_5282176000_cubes.png) |
| 11 | 6.010600 | 40000.0 | 156.2500 | 208 | 208 | 41 | 0.442 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_11_z3846784000_3846809600_cubes.png) |
| 12 | 11.812000 | 40000.0 | 156.2500 | 354 | 354 | 40 | 0.209 | [png](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_12_z7559680000_7559705600_cubes.png) |

### Window 1

![window 1](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_01_z7014630400_7014656000_cubes.png)

### Window 2

![window 2](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_02_z5084390400_5084416000_cubes.png)

### Window 3

![window 3](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_03_z3836620800_3836646400_cubes.png)

### Window 4

![window 4](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_04_z10665625600_10665651200_cubes.png)

### Window 5

![window 5](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_05_z4166476800_4166502400_cubes.png)

### Window 6

![window 6](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_06_z4177305600_4177331200_cubes.png)

### Window 7

![window 7](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_07_z4435404800_4435430400_cubes.png)

### Window 8

![window 8](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_08_z5449625600_5449651200_cubes.png)

### Window 9

![window 9](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_09_z3856332800_3856358400_cubes.png)

### Window 10

![window 10](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_10_z5282150400_5282176000_cubes.png)

### Window 11

![window 11](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_11_z3846784000_3846809600_cubes.png)

### Window 12

![window 12](assets/native_grid_dbscan_fine_voxel_like_crowded_08_7293_many_particles_100x_time_fixed_z/whole_file_window_12_z7559680000_7559705600_cubes.png)

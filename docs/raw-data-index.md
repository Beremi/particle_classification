# Raw Data Index

Generated from the gitignored raw data archive/extraction. Large `.t3pa` files are not tracked.

## Overall

- Files: 1,422
- Total bytes: 1,426,791,071 (1.3 GB)
- Total `.t3pa` hit rows: 44,725,206
- Extensions: `.t3pa` 711, `.t3pa.info` 711

## Folder Summary

| Folder | Files | `.t3pa` files | `.info` files | Bytes | `.t3pa` rows | Acquisition start range | Largest files |
|---|---:|---:|---:|---:|---:|---|---|
| `08_thu_proton_daily_batch` | 736 | 368 | 368 | 41.4 MB | 1,390,014 | Thu Feb  8 04:32:43.700999 2024 to Thu Feb  8 06:36:45.892999 2024 | `toa_tot__r0000007293.t3pa` (14.6 MB)<br>`toa_tot__r0000007296.t3pa` (8.3 MB)<br>`toa_tot__r0000007295.t3pa` (6.7 MB)<br>`toa_tot__r0000007332.t3pa` (3.2 MB)<br>`toa_tot__r0000007294.t3pa` (3.2 MB) |
| `13_tue_proton_daily_batch` | 458 | 229 | 229 | 51.9 MB | 1,757,135 | Tue Feb 13 05:34:46.578999 2024 to Tue Feb 13 06:51:47.766999 2024 | `toa_tot__r0000028554.t3pa` (9.6 MB)<br>`toa_tot__r0000028555.t3pa` (7.9 MB)<br>`toa_tot__r0000028552.t3pa` (7.3 MB)<br>`toa_tot__r0000028553.t3pa` (6.1 MB)<br>`toa_tot__r0000028495.t3pa` (5.6 MB) |
| `D05` | 108 | 54 | 54 | 255.7 MB | 8,570,737 | Sat Mar 14 11:57:41.734999 2026 to Sat Mar 14 12:16:06.667999 2026 | `tot_toa__r0000000052.t3pa` (43.4 MB)<br>`tot_toa__r0000000028.t3pa` (40.1 MB)<br>`tot_toa__r0000000046.t3pa` (29.8 MB)<br>`tot_toa__r0000000027.t3pa` (29.0 MB)<br>`tot_toa__r0000000038.t3pa` (24.1 MB) |
| `F08` | 108 | 54 | 54 | 462.0 MB | 15,474,240 | Sat Mar 14 11:57:39.818000 2026 to Sat Mar 14 12:16:09.627000 2026 | `tot_toa__r0000000052.t3pa` (74.6 MB)<br>`tot_toa__r0000000028.t3pa` (70.2 MB)<br>`tot_toa__r0000000046.t3pa` (51.9 MB)<br>`tot_toa__r0000000027.t3pa` (50.7 MB)<br>`tot_toa__r0000000038.t3pa` (42.0 MB) |
| `data I05` | 6 | 3 | 3 | 438.5 MB | 13,947,284 | Wed Mar  3 11:59:03.332054 2021 to Wed Mar  3 12:01:08.801593 2021 | `sync__I05-W0044_r001.t3pa` (198.6 MB)<br>`sync__I05-W0044_r000.t3pa` (174.0 MB)<br>`sync__I05-W0044_r002.t3pa` (66.0 MB)<br>`sync__I05-W0044_r000.t3pa.info` (944 B)<br>`sync__I05-W0044_r001.t3pa.info` (944 B) |
| `data M07` | 6 | 3 | 3 | 111.1 MB | 3,585,796 | Wed Mar  3 11:59:03.593418 2021 to Wed Mar  3 12:01:08.984111 2021 | `sync__M07-W0044_r001.t3pa` (50.6 MB)<br>`sync__M07-W0044_r000.t3pa` (43.9 MB)<br>`sync__M07-W0044_r002.t3pa` (16.6 MB)<br>`sync__M07-W0044_r000.t3pa.info` (944 B)<br>`sync__M07-W0044_r001.t3pa.info` (944 B) |

## Metadata Notes

- `.t3pa` files are tab-separated hit tables with `Index`, `Matrix Index`, `ToA`, `ToT`, `FToA`, and `Overflow` columns.
- `Matrix Index` is decoded as row-major `x = index % 256`, `y = index // 256`.
- `.t3pa.info` sidecars contain acquisition metadata such as chipboard ID, high voltage, acquisition duration, threshold, Pixet version, and start time.
- The first energy-like feature for NN experiments is `log1p(ToT)` until calibrated energy documentation is added.
- `theta_xy` derived from candidates is pose metadata only and must not be used as a classification or clustering feature.

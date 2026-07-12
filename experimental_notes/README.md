# Experimental notes archive

This directory preserves experiment reports, diagnostic galleries, and design
notes that support the compact [core documentation](../docs/README.md).

These files are historical evidence, not a single current specification.
Results may be superseded by later timestamp corrections, rebuilt particle
shards, changed splits, or newer model choices. Check the status at the top of
an individual report and prefer the core docs plus current code for operation.

## Background and data interpretation

- [`background/`](background/) — classification taxonomy, DBSCAN concepts,
  cluster features, the human-gold labeling schema, and archived technical
  report sources/reference PDFs.
- [`data/`](data/) — focused raw-feature investigations such as ToT/energy
  histograms.

## Phase 1 separation

- [`dbscan/final_eps5/`](dbscan/final_eps5/) — evidence for the active native
  grid DBSCAN eps5 baseline, including speed and visual checks.
- [`dbscan/alternatives/`](dbscan/alternatives/) — parameterizations and
  continuity variants evaluated around the selected baseline.
- [`dbscan/historical/`](dbscan/historical/) — older teachers, audits, and
  approaches retained to explain why they were replaced.
- [`phase1_neural/`](phase1_neural/) — archived EdgeTrackNet/neural separator
  experiments. These are not the active Phase 1 implementation.

## Phase 2 autoencoders

- [`autoencoders/voxel/baseline/`](autoencoders/voxel/baseline/) — voxel sizing,
  cache design, and initial reconstruction experiments.
- [`autoencoders/voxel/simple_z8/`](autoencoders/voxel/simple_z8/) — the compact
  direct z8 voxel training sequence and galleries.
- [`autoencoders/voxel/canonical/`](autoencoders/voxel/canonical/) — canonical
  transform-aware voxel experiments.
- [`autoencoders/voxel/compressed/`](autoencoders/voxel/compressed/) — capacity
  and objective studies on compressed caches.
- [`autoencoders/path/`](autoencoders/path/) — ordered-path, pose-separated, and
  structured-transform investigations.
- [`autoencoders/point_set_legacy/`](autoencoders/point_set_legacy/) — earlier
  point-set reconstruction and embedding work.
- [`autoencoders/latent_analysis/`](autoencoders/latent_analysis/) — latent
  grouping, nearest-pair, PCA, and subgroup diagnostics.

## Generated assets

`assets/` contains images and tables referenced by reports. An asset documents
a particular run; it should not be treated as a reproducible result without the
corresponding report, configuration, source manifest, and checkpoint metadata.
Some archived CSV/JSON payloads intentionally retain their historical
`docs/assets/...` strings; resolve that prefix as `experimental_notes/assets/...`
when replaying old path-based tooling.

For the current operational summary, read [raw data](../docs/raw-data.md),
[DBSCAN](../docs/dbscan.md), and [autoencoders](../docs/autoencoder.md).

# Particle Classification from Sparse Timepix Detector Hits

This repository is now a runnable implementation and testing ground for neural-network particle-candidate classification. The central design rule is **XY invariance**: detector-plane rotation is computed and stored as metadata, but it is not allowed to define a particle morphology class.

## Current Status

- Raw data archive extracted locally to gitignored `local_data/raw/`
- Raw data index tracked in [`docs/raw-data-index.md`](docs/raw-data-index.md) and [`data/raw_data_index.csv`](data/raw_data_index.csv)
- Primary package: [`src/particle_classification`](src/particle_classification)
- Legacy visualization package kept importable as [`src/particle_viz`](src/particle_viz)
- Updated English report: [`particle_nn_report_updated/particle_nn_report.tex`](particle_nn_report_updated/particle_nn_report.tex)
- First NN baseline: compact PointNet/DeepSets-style `XYInvariantParticleNet`

## Design Contract

Candidate particles are represented as variable-size point sets:

```text
P = {(x_i, y_i, t_i, e_i)}
```

The class embedding uses shape, energy density, scale, time pitch, and energy. The detector-plane azimuth `theta_xy` and its reliability `q_theta` are saved separately.

Do not cluster or classify on a descriptor that includes `theta_xy`.

## Repository Layout

```text
.
├── configs/baseline.yaml
├── data/
│   ├── matrix_dump_0001.txt
│   └── raw_data_index.csv
├── docs/
│   ├── raw-data-index.md
│   └── ...
├── particle_nn_report_updated/
├── src/
│   ├── particle_classification/
│   └── particle_viz/
└── tests/
```

Large local data and generated experiment outputs live under `local_data/` and are ignored by git.

## Commands

Install in editable mode:

```bash
python -m pip install -e .
```

Extract raw data:

```bash
particle-extract-raw raw_data.zip --dest local_data/raw
```

Regenerate the raw data index:

```bash
particle-data-index --input raw_data.zip --markdown docs/raw-data-index.md --csv data/raw_data_index.csv
```

Build a lightweight raw candidate/source table:

```bash
particle-build-candidates --input local_data/raw --out local_data/processed/candidates.parquet
```

Tune 3D DBSCAN on a fixed stratified sample:

```bash
particle-tune-dbscan --input local_data/raw --out local_data/processed/dbscan_tuning
```

Build per-file particle NPZ shards:

```bash
particle-build-particles \
  --input local_data/raw \
  --params local_data/processed/dbscan_tuning/best_params.json \
  --out local_data/processed/particles
```

Build pass-1 EdgeTrackNet pseudo-label windows:

```bash
particle-build-edge-training-set \
  --input local_data/processed/particles \
  --params local_data/processed/dbscan_tuning/best_params.json \
  --out local_data/processed/pass1_edge_dataset
```

Build controlled shifted/mixed windows from real particle templates plus procedural particles:

```bash
particle-build-edge-mixed-set \
  --input local_data/processed/particles \
  --params local_data/processed/dbscan_tuning/best_params.json \
  --out local_data/processed/pass1_edge_mixed_dataset \
  --windows 4000
```

Run the preliminary DBSCAN-replacement experiment:

```bash
particle-run-edge-experiment \
  --particles local_data/processed/particles \
  --params local_data/processed/dbscan_tuning/best_params.json \
  --dataset-out local_data/processed/pass1_edge_dataset \
  --experiment-out local_data/experiments/edge_tracknet_long
```

Run the longer target-seeking EdgeTrackNet replacement search with plateau LR and validation threshold sweep:

```bash
particle-run-edge-replacement-search \
  --particles local_data/processed/particles \
  --params local_data/processed/dbscan_tuning/best_params.json \
  --dataset-out local_data/processed/pass1_edge_mixed_dataset \
  --experiment-out local_data/experiments/edge_tracknet_replacement_search \
  --windows 4000 \
  --steps 5000 \
  --min-steps 1000 \
  --message-passing-steps 2
```

Run the minimal XY-invariance training smoke test:

```bash
particle-train-baseline --config configs/baseline.yaml
```

## Modeling Roadmap

1. Deterministic XY-invariant descriptors from weighted PCA and energy-density profiles.
2. DBSCAN as an interpretable candidate generator and bootstrap baseline.
3. `XYInvariantParticleNet` trained with rotation-invariance and profile losses.
4. Optional stronger point models: EdgeConv, GravNet-style blocks, point transformers.
5. Learned clustering or object condensation once simulation or hit-level truth exists.

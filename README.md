# Particle Classification from Sparse Timepix Detector Hits

This repository is now a runnable implementation and testing ground for neural-network particle-candidate classification. The central design rule is **XY invariance**: detector-plane rotation is computed and stored as metadata, but it is not allowed to define a particle morphology class.

## Current Status

- Raw data archive extracted locally to gitignored `local_data/raw/`
- Raw data index tracked in [`docs/raw-data-index.md`](docs/raw-data-index.md) and [`data/raw_data_index.csv`](data/raw_data_index.csv)
- Primary package: [`src/particle_classification`](src/particle_classification)
- Legacy visualization package kept importable as [`src/particle_viz`](src/particle_viz)
- Updated English report: [`particle_nn_report_updated/particle_nn_report.tex`](particle_nn_report_updated/particle_nn_report.tex)
- First NN baseline: compact PointNet/DeepSets-style `XYInvariantParticleNet`
- Phase 1 DBSCAN replacement: leakage-safe `EdgeTrackNetTiny` edge model with source-grouped splits, versioned `dbscan_v001` teacher metadata, 11-feature hit schema, multi-scale graphs, bridge-safe readout, focal edge loss, and bucketed evaluation.
- Latest Phase 1 hardening run note: [`docs/phase1-hardening-v001-results.md`](docs/phase1-hardening-v001-results.md)

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
├── configs/
│   ├── baseline.yaml
│   └── teachers/dbscan_v001.json
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
particle-data-index --input local_data/raw --markdown docs/raw-data-index.md --csv data/raw_data_index.csv
```

Build a lightweight raw candidate/source table:

```bash
particle-build-candidates --input local_data/raw --out local_data/processed/candidates.parquet
```

Tune 3D DBSCAN on a fixed stratified sample:

```bash
particle-tune-dbscan \
  --input local_data/raw \
  --index data/raw_data_index.csv \
  --out local_data/processed/dbscan_tuning_v001
```

Build per-file particle NPZ shards:

```bash
particle-build-particles \
  --input local_data/raw \
  --index data/raw_data_index.csv \
  --params configs/teachers/dbscan_v001.json \
  --out local_data/processed/particles_dbscan_v001 \
  --skip-existing
```

Build leakage-safe pass-1 EdgeTrackNet pseudo-label windows:

```bash
particle-build-edge-training-set \
  --input local_data/processed/particles_dbscan_v001 \
  --params configs/teachers/dbscan_v001.json \
  --out local_data/processed/pass1_edge_real_stable_v001 \
  --max-windows 8000 \
  --window-size 2048 \
  --window-overlap 512 \
  --min-stability-ari 0.75 \
  --split-strategy group-source \
  --teacher-name dbscan_v001
```

Build controlled shifted/mixed windows from real particle templates plus procedural particles:

```bash
particle-build-edge-mixed-set \
  --input local_data/processed/particles_dbscan_v001 \
  --params configs/teachers/dbscan_v001.json \
  --out local_data/processed/pass1_edge_mixed_hard_v001 \
  --windows 12000 \
  --synthetic-fraction 0.50 \
  --hard-fraction 0.70 \
  --split-strategy group-source \
  --teacher-name dbscan_v001
```

Train the Phase 1 edge model:

```bash
particle-train-edge-tracknet \
  --manifest local_data/processed/pass1_edge_real_stable_v001/manifest.csv \
  --out local_data/experiments/phase1_A_real_stable \
  --steps 6000 \
  --min-steps 1500 \
  --batch-size 4 \
  --hidden-dim 128 \
  --edge-hidden-dim 128 \
  --message-passing-steps 2 \
  --edge-loss focal \
  --embedding-loss-weight 0.0 \
  --device cuda
```

Evaluate one checkpoint against real and hard-mixed test splits:

```bash
particle-evaluate-phase1 \
  --checkpoint local_data/experiments/phase1_A_real_stable/edge_tracknet_tiny.pt \
  --normalization local_data/processed/pass1_edge_real_stable_v001/normalization.json \
  --manifest local_data/processed/pass1_edge_real_stable_v001/manifest.csv \
  --manifest local_data/processed/pass1_edge_mixed_hard_v001/manifest.csv \
  --out local_data/experiments/phase1_A_real_stable/evaluation \
  --device cuda
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

Future hand-corrected validation labels should follow [`docs/human-gold-schema.md`](docs/human-gold-schema.md). The evaluator keeps `teacher_dbscan`, `synthetic_truth`, and `human_gold` metrics separated.

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

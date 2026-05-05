# Particle Classification from Sparse Timepix Detector Hits

This repository is now a runnable implementation and testing ground for Timepix particle-candidate extraction and classification. The Phase 1 particle-separation baseline is a custom native 3D DBSCAN implementation, while neural-network separator attempts are archived as research notes for later learned alternatives. The central design rule for downstream classification remains **XY invariance**: detector-plane rotation is computed and stored as metadata, but it is not allowed to define a particle morphology class.

## Current Status

- Raw data archive extracted locally to gitignored `local_data/raw/`
- Raw data index tracked in [`docs/raw-data-index.md`](docs/raw-data-index.md) and [`data/raw_data_index.csv`](data/raw_data_index.csv)
- Primary package: [`src/particle_classification`](src/particle_classification)
- Legacy visualization package kept importable as [`src/particle_viz`](src/particle_viz)
- Updated English report: [`particle_nn_report_updated/particle_nn_report.tex`](particle_nn_report_updated/particle_nn_report.tex)
- First NN baseline: compact PointNet/DeepSets-style `XYInvariantParticleNet`
- Final Phase 1 separator baseline: custom C/OpenMP `native-grid-dbscan` with corrected fine time and `configs/teachers/dbscan_phase1_native_eps5_v001.json`, documented in [`docs/phase1-native-grid-baseline.md`](docs/phase1-native-grid-baseline.md)
- Important DBSCAN audit: old `dbscan_v001` labels match the native backend exactly, but are not trusted particle labels because quality checks found file-scale merged components; see [`docs/dbscan-teacher-audit.md`](docs/dbscan-teacher-audit.md)
- 3D clustering speed report: [`docs/clustering-speed-report.md`](docs/clustering-speed-report.md)
- Phase 2 particle embedding/clustering report: [`docs/phase2-particle-embedding-report.md`](docs/phase2-particle-embedding-report.md)
- Archived Phase 1 NN reports: [`docs/phase1-nn-full-report.md`](docs/phase1-nn-full-report.md), [`docs/phase1-hardening-v001-results.md`](docs/phase1-hardening-v001-results.md), and [`docs/phase1-dbscan-replacement-report.md`](docs/phase1-dbscan-replacement-report.md)

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
│   └── teachers/
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

Build per-file particle NPZ shards with the active Phase 1 baseline:

```bash
particle-build-particles \
  --input local_data/raw \
  --index data/raw_data_index.csv \
  --params configs/teachers/dbscan_phase1_native_eps5_v001.json \
  --out local_data/processed/particles_aligned_time_eps5_v001 \
  --backend native-grid-dbscan \
  --threads 32
```

Benchmark exact and experimental 3D clustering backends:

```bash
particle-benchmark-clustering \
  --input local_data/raw \
  --params configs/teachers/dbscan_phase1_native_eps5_v001.json \
  --out local_data/benchmarks/clustering_native_v001 \
  --cases largest,slowest_per_hit,max_particles \
  --backend ckdtree-pairs \
  --backend numba-grid-dbscan \
  --backend native-grid-dbscan \
  --backend numba-stream-grid-linker \
  --backend native-stream-grid-linker \
  --threads 32 \
  --repeat-runs 1 \
  --toa-tick-ns 25
```

Build the Phase 2 variable-hit particle embedding dataset from native Phase 1 shards:

```bash
particle-build-phase2-dataset \
  --input local_data/processed/particles_aligned_time_eps5_v001 \
  --out local_data/processed/phase2_particles_v001 \
  --params configs/teachers/dbscan_phase1_native_eps5_v001.json \
  --max-points 512 \
  --views-per-large-particle 4 \
  --source-backend native-grid-dbscan \
  --teacher-name dbscan_phase1_native_eps5_v001
```

Run the Phase 2 architecture/objective sweep and clustering evaluation:

```bash
particle-train-phase2-sweep \
  --dataset local_data/processed/phase2_particles_v001 \
  --out local_data/experiments/phase2_particle_sweep_v001 \
  --budget overnight \
  --device cuda

particle-evaluate-phase2 \
  --dataset local_data/processed/phase2_particles_v001 \
  --experiment local_data/experiments/phase2_particle_sweep_v001 \
  --out local_data/experiments/phase2_particle_sweep_v001/evaluation \
  --device cuda

particle-generate-phase2-report \
  --dataset local_data/processed/phase2_particles_v001 \
  --experiment local_data/experiments/phase2_particle_sweep_v001 \
  --evaluation local_data/experiments/phase2_particle_sweep_v001/evaluation \
  --out docs/phase2-particle-embedding-report.md
```

The neural Phase 1 separator attempts are archived for now. The commands below
remain available for reproducing those experiments, but they are not the active
Phase 1 baseline.

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

Build and fine-tune a real/hard curriculum from the Stage A checkpoint:

```bash
particle-build-edge-curriculum-set \
  --real-manifest local_data/processed/pass1_edge_real_stable_v001/manifest.csv \
  --mixed-manifest local_data/processed/pass1_edge_mixed_hard_v001/manifest.csv \
  --normalization local_data/processed/pass1_edge_real_stable_v001/normalization.json \
  --out local_data/processed/pass1_edge_curriculum_v001 \
  --train-real-ratio 0.50 \
  --val-real-ratio 0.50 \
  --test-real-ratio 0.50

particle-train-edge-tracknet \
  --manifest local_data/processed/pass1_edge_curriculum_v001/manifest.csv \
  --out local_data/experiments/phase1_C_curriculum_finetune \
  --init-checkpoint local_data/experiments/phase1_A_real_stable/edge_tracknet_tiny.pt \
  --steps 10000 \
  --min-steps 3000 \
  --batch-size 4 \
  --learning-rate 0.0002 \
  --hidden-dim 128 \
  --edge-hidden-dim 128 \
  --message-passing-steps 2 \
  --edge-loss focal \
  --embedding-loss-weight 0.03 \
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
2. `native-grid-dbscan` as the active real-time Phase 1 particle separator.
3. Phase 2 particle-level embedding on native-generated shards using variable-hit point encoders and descriptor summaries.
4. Unsupervised family discovery with HDBSCAN/k-means/GMM/DEC/VaDE, with cluster IDs named only after inspection or external truth.
5. Supervised imitators of frozen discovered families for deployment once a useful family map is selected.

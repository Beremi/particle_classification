# Phase 1 Reliability Hardening v001 Results

Run date: 2026-05-03.

## Teacher and Data

`dbscan_v001` was tuned from the current `local_data/raw` snapshot and written to
`configs/teachers/dbscan_v001.json`.

| Item | Value |
|---|---:|
| Raw `.t3pa` files | 711 |
| Raw hits | 44,725,206 |
| Tuned `eps` | 5.0 |
| Tuned `min_samples` | 3 |
| Tuned `time_scale` | 15,000,000 |
| Tuning score | 0.5637 |

The full DBSCAN particle extraction completed for all 711 files under
`local_data/processed/particles_dbscan_v001/`.

| Output | Value |
|---|---:|
| Particle shard files | 711 |
| Status `ok` | 711 |
| Total particles | 98,341 |
| Total DBSCAN noise hits | 41,489 |
| Output size | 369 MB |

## Edge Datasets

Both datasets use source-grouped splits. The maximum number of splits per
`split_group` is `1`, so there is no source leakage by construction.

| Dataset | Label Source | Windows | Train | Val | Test | Hits | Edges |
|---|---|---:|---:|---:|---:|---:|---:|
| `pass1_edge_real_stable_v001` | `teacher_dbscan` | 1,043 | 704 | 210 | 129 | 942,289 | 11,304,503 |
| `pass1_edge_mixed_hard_v001` | `synthetic_truth` | 12,000 | 8,409 | 1,670 | 1,921 | 2,361,190 | 61,512,674 |

Both datasets use schema `phase1_nodes_v2`:

- node features: 11
- edge attributes: 14
- graph types: local, medium, time-neighbor, same-pixel
- normalized `features` plus preserved `raw_features`

## Training Results

Stage A trained on source-separated real stable DBSCAN pseudo-labels. Stage B
trained on the hard mixed synthetic/template stress set.

Each row below evaluates one neural-network checkpoint. DBSCAN appears here only
as `teacher_dbscan`, meaning "the DBSCAN labels used as the scoring reference."
It is not a separate model being trained in the table. For `synthetic_truth`,
the scoring reference is the known particle identity from generated/mixed test
windows.

| Neural Network Checkpoint | Eval Set | Scoring Reference | ARI | Pairwise F1 | Split | Merge | Object Acc. | Energy Err. |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `phase1_A_real_stable` | real stable test | `teacher_dbscan` | 0.9233 | 0.9518 | 0.0365 | 0.0024 | 0.9830 | 0.0173 |
| `phase1_A_real_stable` | hard mixed test subset | `synthetic_truth` | 0.3130 | 0.5388 | 0.0201 | 0.7140 | 0.9967 | 0.0029 |
| `phase1_B_hard_mixed` | real stable test | `teacher_dbscan` | 0.5198 | 0.8003 | 0.1771 | 0.0225 | 0.7486 | 0.6160 |
| `phase1_B_hard_mixed` | hard mixed test subset | `synthetic_truth` | 0.6209 | 0.6959 | 0.3465 | 0.1590 | 0.9990 | 0.0009 |

Latency from the Phase 1 evaluator on CUDA:

| Checkpoint | p50 | p95 | p99 |
|---|---:|---:|---:|
| `phase1_A_real_stable` | 0.47 ms | 0.68 ms | 0.94 ms |
| `phase1_B_hard_mixed` | 0.56 ms | 0.82 ms | 0.96 ms |

## Acceptance Status

Real stable DBSCAN imitation is successful for this pass:

- ARI target `>= 0.85`: passed.
- Pairwise F1 target `>= 0.95`: passed in the normalized evaluator run.
- Split target `<= 0.05`: passed.
- Merge target `<= 0.05`: passed.

Hard mixed robustness is not solved:

- ARI target `>= 0.75`: failed.
- Pairwise F1 target `>= 0.85`: failed.
- Split target `<= 0.15`: failed.
- Merge target `<= 0.15`: near threshold in the Stage B full-test summary, but slightly above in the 512-window evaluator subset.

The main remaining failure is still hard-overlap separation. Stage A merges hard
mixed cases aggressively. Stage B improves hard mixed ARI but fragments too much
and does not transfer back to real stable windows.

## Artifacts

- Teacher tuning: `local_data/processed/dbscan_tuning_v001/`
- Particle shards: `local_data/processed/particles_dbscan_v001/`
- Real stable dataset: `local_data/processed/pass1_edge_real_stable_v001/`
- Hard mixed dataset: `local_data/processed/pass1_edge_mixed_hard_v001/`
- Stage A experiment: `local_data/experiments/phase1_A_real_stable/`
- Stage B experiment: `local_data/experiments/phase1_B_hard_mixed/`
- Evaluator outputs: `evaluation/summary.json`, `metrics_by_bucket.csv`,
  `object_iou_metrics.csv`, `edge_pr_curve.csv`, `threshold_sweep.csv`,
  `latency_cpu.json`, and `failure_examples/*.png`

## Next Work

The next useful pass should focus on the hard mixed failure mode rather than
larger model capacity:

- train a combined curriculum instead of pure Stage A or pure Stage B;
- rebalance hard mixed batches toward close negative edges;
- add explicit generator metadata buckets to the evaluator report;
- tune bridge pruning separately for hard crossing/parallel cases;
- consider checkpoint fine-tuning from Stage A into hard mixed, instead of
  training Stage B from scratch.

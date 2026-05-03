from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from particle_classification.models import EdgeTrackNetTiny, EdgeTrackNetTinyConfig


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "phase1-nn-full-report.md"
ASSETS = ROOT / "docs" / "assets" / "phase1_nn_full_report"
EXPERIMENTS = ROOT / "local_data" / "experiments"


RUNS = {
    "A": {
        "name": "phase1_A_real_stable",
        "title": "Stage A real-stable",
        "training": "real stable DBSCAN pseudo-label windows",
    },
    "B": {
        "name": "phase1_B_hard_mixed",
        "title": "Stage B hard-mixed",
        "training": "hard mixed synthetic/template windows",
    },
    "C": {
        "name": "phase1_C_curriculum_finetune",
        "title": "Stage C 50/50 curriculum",
        "training": "Stage A fine-tune on 50/50 real and hard curriculum",
    },
    "C2": {
        "name": "phase1_C2_conservative_curriculum",
        "title": "Stage C2 65/35 curriculum",
        "training": "Stage A fine-tune on 65/35 real-heavy curriculum",
    },
}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for path in ASSETS.glob("*.png"):
        path.unlink()
    summaries = {key: load_json(EXPERIMENTS / run["name"] / "summary.json") for key, run in RUNS.items()}
    evals = {key: load_json(EXPERIMENTS / run["name"] / "evaluation" / "summary.json") for key, run in RUNS.items()}
    checkpoints = {key: load_checkpoint_info(EXPERIMENTS / run["name"] / "edge_tracknet_tiny.pt") for key, run in RUNS.items()}
    plot_convergence()
    plot_lr()
    plot_external_metrics(evals)
    copy_failure_examples()
    DOC.write_text(render_report(summaries, evals, checkpoints), encoding="utf-8")
    print(DOC)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint_info(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = EdgeTrackNetTinyConfig(**checkpoint["model_config"])
    model = EdgeTrackNetTiny(config)
    return {
        "config": checkpoint["model_config"],
        "parameters": sum(param.numel() for param in model.parameters()),
        "edge_threshold": checkpoint.get("selected_edge_threshold"),
        "object_threshold": checkpoint.get("selected_object_threshold"),
        "init_checkpoint": checkpoint.get("init_checkpoint", ""),
    }


def metrics_csv(key: str) -> pd.DataFrame:
    path = EXPERIMENTS / RUNS[key]["name"] / "metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_convergence() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, run in RUNS.items():
        df = metrics_csv(key)
        if df.empty:
            continue
        axes[0].plot(df["step"], df["val_ari"], label=key)
        axes[1].plot(df["step"], df["val_pairwise_f1"], label=key)
    axes[0].set_title("Validation ARI")
    axes[1].set_title("Validation Pairwise F1")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        ax.legend(title="stage")
    fig.savefig(ASSETS / "convergence.png", dpi=150)
    plt.close(fig)


def plot_lr() -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for key in ["C", "C2"]:
        df = metrics_csv(key)
        if df.empty:
            continue
        ax.plot(df["step"], df["lr"], label=key)
    ax.set_title("Curriculum LR progression")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(title="stage")
    fig.savefig(ASSETS / "lr_progression.png", dpi=150)
    plt.close(fig)


def plot_external_metrics(evals: dict[str, dict]) -> None:
    rows = []
    for key, data in evals.items():
        for row in data["rows"]:
            rows.append(
                {
                    "stage": key,
                    "set": "real" if row["label_source"] == "teacher_dbscan" else "hard",
                    "ari": row["ari"],
                    "pairwise_f1": row["pairwise_f1"],
                    "split_rate": row["split_rate"],
                    "merge_rate": row["merge_rate"],
                }
            )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), ["ari", "pairwise_f1", "split_rate", "merge_rate"]):
        pivot = df.pivot(index="stage", columns="set", values=metric).loc[["A", "B", "C", "C2"]]
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("")
    fig.savefig(ASSETS / "external_metric_comparison.png", dpi=150)
    plt.close(fig)


def copy_failure_examples() -> None:
    for key in ["A", "B", "C", "C2"]:
        source_dir = EXPERIMENTS / RUNS[key]["name"] / "evaluation" / "failure_examples"
        if not source_dir.exists():
            continue
        for name in ["manifest_00_00.png", "manifest_01_00.png"]:
            path = source_dir / name
            if path.exists():
                shutil.copyfile(path, ASSETS / f"{key}_{name}")


def external_table(evals: dict[str, dict]) -> str:
    lines = [
        "| Stage | Eval set | Reference labels | ARI | Pairwise F1 | Split | Merge | Object acc. | Energy err. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["A", "B", "C", "C2"]:
        for row in evals[key]["rows"]:
            eval_set = "real stable" if row["label_source"] == "teacher_dbscan" else "hard mixed"
            lines.append(
                "| {stage} | {eval_set} | `{label}` | {ari:.4f} | {f1:.4f} | {split:.4f} | "
                "{merge:.4f} | {obj:.4f} | {energy:.4f} |".format(
                    stage=key,
                    eval_set=eval_set,
                    label=row["label_source"],
                    ari=row["ari"],
                    f1=row["pairwise_f1"],
                    split=row["split_rate"],
                    merge=row["merge_rate"],
                    obj=row["object_accuracy"],
                    energy=row["energy_error"],
                )
            )
    return "\n".join(lines)


def training_table(summaries: dict[str, dict], checkpoints: dict[str, dict]) -> str:
    lines = [
        "| Stage | Trained on | Init | Steps | Best step | Edge/object threshold | Params |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for key in ["A", "B", "C", "C2"]:
        summary = summaries[key]
        info = checkpoints[key]
        init = "Stage A" if info["init_checkpoint"] else "scratch"
        lines.append(
            "| {key} | {training} | {init} | {steps} | {best} | {edge}/{obj} | {params:,} |".format(
                key=key,
                training=RUNS[key]["training"],
                init=init,
                steps=summary["steps"],
                best=summary["best_step"],
                edge=info["edge_threshold"],
                obj=info["object_threshold"],
                params=info["parameters"],
            )
        )
    return "\n".join(lines)


def render_report(summaries: dict[str, dict], evals: dict[str, dict], checkpoints: dict[str, dict]) -> str:
    config = checkpoints["A"]["config"]
    return f"""# Phase 1 Full NN Report: DBSCAN Replacement

Generated from local experiment artifacts on 2026-05-03.

## Problem Statement

The Phase 1 goal is to replace the first 3D DBSCAN particle-grouping pass with a neural network that operates directly on variable-size Timepix hit lists. A particle candidate is represented as a sequence of hits containing detector position, time, and energy-like charge information:

```text
(x, y, time, energy)
```

DBSCAN remains the teacher for stable real data, not the target architecture. The NN must preserve every hit and produce particle IDs that can reconstruct ragged particle sequences. Energy is preserved for classification and diagnostics but is not used as a geometric clustering coordinate.

## Data Flow and Label Sources

Raw `.t3pa` files are parsed into hit arrays, then clustered by the frozen `dbscan_v001` teacher into NPZ particle shards. From those shards the pipeline builds training windows:

```text
raw .t3pa
  -> DBSCAN teacher particle shards
  -> real stable DBSCAN windows, label_source=teacher_dbscan
  -> hard mixed synthetic/template windows, label_source=synthetic_truth
  -> curriculum windows for Stage C/C2 fine-tuning
```

The scoring reference column in result tables is the label source. `teacher_dbscan` means the NN is compared with DBSCAN pseudo-labels. `synthetic_truth` means the NN is compared with known generated particle IDs. Those labels are not separate neural networks.

| Dataset | Windows | Train | Val | Test | Notes |
|---|---:|---:|---:|---:|---|
| real stable DBSCAN v001 | 1,043 | 704 | 210 | 129 | source-grouped real windows, stability ARI filtered |
| hard mixed v001 | 12,000 | 8,409 | 1,670 | 1,921 | procedural plus shifted real templates |
| C curriculum 50/50 | 2,086 | 1,408 | 420 | 258 | all real windows plus matched hard windows |
| C2 curriculum 65/35 | 1,604 | 1,083 | 323 | 198 | all real windows plus smaller hard subset |

## Input Window Structure

Each NPZ window stores one ragged graph:

- `features`: normalized node features, shape `[n_hits, 11]`.
- `raw_features`: unnormalized deterministic node features before dataset normalization.
- `edge_index`: directed graph edges, shape `[2, n_edges]`.
- `edge_attr`: deterministic edge features, shape `[n_edges, 14]`.
- `edge_label`: `1` for same particle, `0` for different particles, `-1` for ignored/noise edge.
- `object_label`: per-hit non-noise label.
- `source_particle_id`: reference particle ID per hit, with `-1` for noise.
- `hit_energy`: `log1p(ToT)` energy proxy used for diagnostics and downstream classification.

The 11 node features are:

| Feature | Definition |
|---|---|
| `x_centered` | `(x - 127.5) / 128` |
| `y_centered` | `(y - 127.5) / 128` |
| `t_scaled` | `(ToA - window_ToA_min) / teacher_time_scale` |
| `t_norm` | per-window min-max normalized ToA |
| `log_tot` | `log1p(ToT) / log1p(1023)` |
| `ftoa_norm` | `FToA / 30` |
| `dt_prev_scaled` | clipped previous ToA gap divided by time scale |
| `dt_next_scaled` | clipped next ToA gap divided by time scale |
| `local_density_r2` | neighbor count within radius 2 |
| `local_density_r4` | neighbor count within radius 4 |
| `knn_dist_4` | fourth-nearest neighbor distance in scaled `(x, y, t)` |

The 14 edge attributes are `dx`, `dy`, signed `dt`, `abs_dt`, `dist_xy`, `dist_xyt`, signed `dlogE`, `abs_dlogE`, `logE_ratio`, `same_pixel`, and four edge-type flags. The graph is multi-scale: local radius/kNN edges, medium radius/kNN edges, time-neighbor edges, and same-pixel time-local edges.

## Synthetic and Mixed Data Structure

Hard mixed windows are designed as a stress test, not as an easy training set. They contain:

- shifted real DBSCAN particle templates, preserving hit multiplicity and energy profiles;
- procedural straight, kinked, compact, and dense particle shapes;
- controlled close placement in `(x, y, time)` to create hard negative edges;
- injected noise hits and repeated pixels;
- known `synthetic_truth` labels for every generated particle.

The important difference from real-stable DBSCAN windows is edge balance. Real-stable windows are dominated by easy positive local edges. Hard mixed windows contain many close negative edges, which exposes merge failures in connected-components readout.

## Network Structure

The primary model is `EdgeTrackNetTiny` with this Stage A/C configuration:

| Setting | Value |
|---|---:|
| input_dim | {config["input_dim"]} |
| edge_attr_dim | {config["edge_attr_dim"]} |
| hidden_dim | {config["hidden_dim"]} |
| edge_hidden_dim | {config["edge_hidden_dim"]} |
| message_passing_steps | {config["message_passing_steps"]} |
| dropout | {config["dropout"]} |
| trainable parameters | {checkpoints["A"]["parameters"]:,} |

The forward pass is:

1. Encode each hit from 11 input features into a 128-dimensional hidden state with `Linear -> LayerNorm -> SiLU -> Dropout -> Linear -> LayerNorm -> SiLU`.
2. For each message-passing step, build an edge message from `[h_src, h_dst, h_src - h_dst, edge_attr]`.
3. Mean-aggregate messages into destination hits and apply a residual node update.
4. Predict one same-particle edge logit per directed edge.
5. Predict one objectness logit per hit.
6. Convert edge probabilities into particle IDs with bridge-safe connected components.

No custom weight initialization is used. PyTorch `Linear` layers use their default Kaiming-uniform reset; `LayerNorm` starts with scale one and bias zero. Dataset normalization is stored in `normalization.json`, and checkpoints embed the normalization used for evaluation.

## Training and Losses

Training uses AdamW, plateau LR scheduling, focal edge loss, weighted objectness BCE, and optional node embedding metric loss. Stage A uses only real stable DBSCAN pseudo-labels. Stage B uses hard mixed data from scratch. Stages C and C2 fine-tune from the Stage A checkpoint.

{training_table(summaries, checkpoints)}

![Convergence](assets/phase1_nn_full_report/convergence.png)

![LR progression](assets/phase1_nn_full_report/lr_progression.png)

## External Evaluation

The table below evaluates each neural checkpoint against the original real-stable and hard-mixed test manifests. These are separate scoring references and must not be averaged together.

{external_table(evals)}

![External metric comparison](assets/phase1_nn_full_report/external_metric_comparison.png)

## Validation Images

Representative examples are copied from the Phase 1 evaluator. The left subplot is the reference label set and the right subplot is the NN prediction.

| Stage | Real-stable example | Hard-mixed example |
|---|---|---|
| A | ![A real example](assets/phase1_nn_full_report/A_manifest_00_00.png) | ![A hard example](assets/phase1_nn_full_report/A_manifest_01_00.png) |
| B | ![B real example](assets/phase1_nn_full_report/B_manifest_00_00.png) | ![B hard example](assets/phase1_nn_full_report/B_manifest_01_00.png) |
| C | ![C real example](assets/phase1_nn_full_report/C_manifest_00_00.png) | ![C hard example](assets/phase1_nn_full_report/C_manifest_01_00.png) |
| C2 | ![C2 real example](assets/phase1_nn_full_report/C2_manifest_00_00.png) | ![C2 hard example](assets/phase1_nn_full_report/C2_manifest_01_00.png) |

Full local evaluator outputs are under `local_data/experiments/*/evaluation/` and include `summary.json`, `metrics_by_bucket.csv`, `object_iou_metrics.csv`, `edge_pr_curve.csv`, `threshold_sweep.csv`, `latency_cpu.json`, and more failure examples.

## Interpretation

Stage A remains the best deployable Phase 1 checkpoint because it is the only checkpoint that clearly satisfies the real-stable DBSCAN replacement gate with the selected checkpoint thresholds. It reaches real-stable ARI above 0.92 and pairwise F1 above 0.95 while keeping split and merge rates below 0.05.

The curriculum attempts are informative but not successful as replacements. Stage C can pass the real-stable gate only with a different external threshold from its checkpoint-selected threshold, and neither C nor C2 improves hard-mixed performance over Stage B. Stage B remains the strongest hard-mixed checkpoint, but it does not transfer back to real-stable data.

The remaining failure is not raw model size. It is the hard-overlap objective and readout: close particles still merge through plausible high-confidence bridge edges, while higher thresholds fragment some tracks. The next pass should focus on threshold calibration by label source, better hard-negative curriculum sampling, explicit generator buckets, and a learned or more conservative component-splitting readout.

## Reproduction Commands

```bash
particle-build-edge-curriculum-set \\
  --real-manifest local_data/processed/pass1_edge_real_stable_v001/manifest.csv \\
  --mixed-manifest local_data/processed/pass1_edge_mixed_hard_v001/manifest.csv \\
  --normalization local_data/processed/pass1_edge_real_stable_v001/normalization.json \\
  --out local_data/processed/pass1_edge_curriculum_v001 \\
  --train-real-ratio 0.50 \\
  --val-real-ratio 0.50 \\
  --test-real-ratio 0.50

particle-train-edge-tracknet \\
  --manifest local_data/processed/pass1_edge_curriculum_v001/manifest.csv \\
  --out local_data/experiments/phase1_C_curriculum_finetune \\
  --init-checkpoint local_data/experiments/phase1_A_real_stable/edge_tracknet_tiny.pt \\
  --steps 10000 \\
  --min-steps 3000 \\
  --batch-size 4 \\
  --learning-rate 0.0002 \\
  --eval-interval 100 \\
  --max-eval-windows 256 \\
  --hidden-dim 128 \\
  --edge-hidden-dim 128 \\
  --message-passing-steps 2 \\
  --edge-loss focal \\
  --embedding-loss-weight 0.03 \\
  --device cuda
```
"""


if __name__ == "__main__":
    main()

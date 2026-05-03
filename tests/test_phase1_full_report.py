import importlib.util
from pathlib import Path


def test_phase1_full_report_render_separates_models_from_references():
    module_path = Path("scripts/generate_phase1_nn_full_report.py")
    spec = importlib.util.spec_from_file_location("phase1_report", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summaries = {key: {"steps": 1, "best_step": 1} for key in ["A", "B", "C", "C2"]}
    checkpoints = {
        key: {
            "config": {
                "input_dim": 11,
                "edge_attr_dim": 14,
                "hidden_dim": 128,
                "edge_hidden_dim": 128,
                "message_passing_steps": 2,
                "dropout": 0.05,
            },
            "parameters": 123,
            "edge_threshold": 0.7,
            "object_threshold": 0.9,
            "init_checkpoint": "" if key in {"A", "B"} else "stage_a.pt",
        }
        for key in ["A", "B", "C", "C2"]
    }
    evals = {
        key: {
            "rows": [
                metric_row("teacher_dbscan", 0.9),
                metric_row("synthetic_truth", 0.5),
            ]
        }
        for key in ["A", "B", "C", "C2"]
    }

    report = module.render_report(summaries, evals, checkpoints)

    assert "# Phase 1 Full NN Report" in report
    assert "`teacher_dbscan` means the NN is compared with DBSCAN pseudo-labels" in report
    assert "Those labels are not separate neural networks" in report
    assert "C curriculum 50/50" in report


def metric_row(label_source: str, value: float) -> dict[str, float | str]:
    return {
        "label_source": label_source,
        "ari": value,
        "pairwise_f1": value,
        "split_rate": 0.01,
        "merge_rate": 0.02,
        "object_accuracy": 0.99,
        "energy_error": 0.01,
    }

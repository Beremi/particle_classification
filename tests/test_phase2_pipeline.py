from particle_classification.data.phase2_dataset import Phase2DatasetConfig, build_phase2_dataset
from particle_classification.phase2 import (
    Phase2TrainConfig,
    evaluate_phase2_experiment,
    generate_phase2_report,
    train_phase2_sweep,
)

from test_phase2_dataset import make_phase2_particle_source


def test_phase2_smoke_train_evaluate_and_report(tmp_path):
    particles = make_phase2_particle_source(tmp_path)
    dataset_dir = tmp_path / "phase2"
    build_phase2_dataset(
        particles,
        dataset_dir,
        config=Phase2DatasetConfig(
            max_points=8,
            large_particle_threshold=10,
            views_per_large_particle=2,
            val_fraction=0.0,
            test_fraction=0.0,
        ),
    )

    experiment_dir = tmp_path / "experiment"
    summary = train_phase2_sweep(
        dataset_dir,
        experiment_dir,
        config=Phase2TrainConfig(
            budget="smoke",
            steps=2,
            min_steps=1,
            eval_interval=1,
            batch_size=2,
            max_eval_items=4,
            run_limit=1,
            amp=False,
        ),
    )
    assert summary["runs"]
    evaluation = evaluate_phase2_experiment(dataset_dir, experiment_dir, experiment_dir / "evaluation", max_items=4)
    assert evaluation["evaluated_rows"] >= 1
    report = generate_phase2_report(
        experiment_dir,
        tmp_path / "phase2-report.md",
        dataset_dir=dataset_dir,
        evaluation_dir=experiment_dir / "evaluation",
    )
    assert report["runs"] == 1
    assert (tmp_path / "phase2-report.md").read_text(encoding="utf-8").startswith("# Phase 2 Particle Embedding Report")

import csv
from pathlib import Path

import numpy as np
import torch

from particle_classification.data.edge_training import (
    EdgeDatasetConfig,
    build_edge_training_set,
    edge_labels,
    make_edge_window,
)
from particle_classification.data.edge_mixing import MixedEdgeDatasetConfig, build_mixed_edge_training_set
from particle_classification.data.particles import DBSCANParticleParams
from particle_classification.edge_training import (
    EdgeTrainConfig,
    collate_edge_windows,
    edge_tracknet_loss,
    model_selection_score,
    train_edge_tracknet,
)
from particle_classification.models import EdgeTrackNetTiny, connected_components_from_edges


def test_edge_window_generation_and_labels():
    window = synthetic_window()
    params = DBSCANParticleParams(eps=2.0, min_samples=2, time_scale=10.0)
    config = EdgeDatasetConfig(k_neighbors=6, radius=120.0, min_hits=1, min_stability_ari=0.0)

    edge_window = make_edge_window(window, params=params, config=config)

    assert edge_window is not None
    assert edge_window["features"].shape == (7, 7)
    assert edge_window["edge_index"].shape[0] == 2
    labels = edge_labels(edge_window["edge_index"], window["hit_particle_id"])
    assert np.any(labels == 1)
    assert np.any(labels == 0)
    assert np.any(labels == -1)
    assert np.all(edge_window["edge_weight"][edge_window["edge_label"] >= 0] > 0)


def test_edge_tracknet_forward_loss_and_components():
    window = synthetic_window()
    params = DBSCANParticleParams(eps=2.0, min_samples=2, time_scale=10.0)
    edge_window = make_edge_window(
        window,
        params=params,
        config=EdgeDatasetConfig(k_neighbors=3, radius=30.0, min_hits=1, min_stability_ari=0.0),
    )
    assert edge_window is not None
    item = {key: value for key, value in edge_window.items() if isinstance(value, np.ndarray)}
    batch = collate_edge_windows([{**item, "path": "synthetic"}])
    model = EdgeTrackNetTiny(input_dim=7, hidden_dim=16, edge_hidden_dim=16, message_passing_steps=1)

    output = model(batch["features"], batch["edge_index"])
    loss, metrics = edge_tracknet_loss(output, batch)
    pred = connected_components_from_edges(
        int(batch["features"].shape[0]),
        batch["edge_index"].numpy(),
        torch.sigmoid(output["edge_logits"]).detach().numpy(),
        torch.sigmoid(output["object_logits"]).detach().numpy(),
    )

    assert output["edge_logits"].shape[0] == batch["edge_index"].shape[1]
    assert torch.isfinite(loss)
    assert metrics["loss"] >= 0.0
    assert pred.shape == (7,)


def test_build_edge_training_set_and_train_smoke(tmp_path):
    particles = tmp_path / "particles"
    particles.mkdir()
    shard = particles / "sample.particles.npz"
    write_synthetic_particle_shard(shard)
    manifest = particles / "manifest.csv"
    manifest.write_text(
        "source_path,output_path,row_count,status\n"
        f"sample.t3pa,{shard.as_posix()},7,ok\n",
        encoding="utf-8",
    )
    params_path = tmp_path / "params.json"
    params_path.write_text('{"eps": 2.0, "min_samples": 2, "time_scale": 10.0}', encoding="utf-8")

    result = build_edge_training_set(
        particles,
        tmp_path / "edge_dataset",
        params_path=params_path,
        config=EdgeDatasetConfig(
            window_size=10,
            window_overlap=0,
                k_neighbors=3,
                radius=30.0,
                min_hits=1,
                min_stability_ari=0.0,
            max_windows=3,
            val_fraction=0.0,
            test_fraction=0.0,
        ),
    )
    train_result = train_edge_tracknet(
        tmp_path / "edge_dataset" / "manifest.csv",
        tmp_path / "experiment",
        config=EdgeTrainConfig(steps=2, batch_size=1, hidden_dim=16, edge_hidden_dim=16, eval_interval=1),
    )

    assert result["windows"] == 1
    assert Path(result["manifest"]).exists()
    assert Path(train_result["checkpoint"]).exists()
    assert Path(tmp_path / "experiment" / "threshold_sweep.csv").exists()
    assert "selected_edge_threshold" in train_result


def test_build_mixed_edge_training_set_from_shifted_templates(tmp_path):
    particles = tmp_path / "particles"
    particles.mkdir()
    shard = particles / "sample.particles.npz"
    write_synthetic_particle_shard(shard)
    manifest = particles / "manifest.csv"
    manifest.write_text(
        "source_path,output_path,row_count,particle_count,validation_warnings,status\n"
        f"sample.t3pa,{shard.as_posix()},7,2,,ok\n",
        encoding="utf-8",
    )
    params_path = tmp_path / "params.json"
    params_path.write_text('{"eps": 2.0, "min_samples": 2, "time_scale": 10.0}', encoding="utf-8")

    result = build_mixed_edge_training_set(
        particles,
        tmp_path / "mixed_edge_dataset",
        params_path=params_path,
        config=MixedEdgeDatasetConfig(
            windows=4,
            synthetic_fraction=0.5,
            hard_fraction=1.0,
            min_particle_hits=2,
            max_source_shards=1,
            min_particles=2,
            max_particles=3,
            k_neighbors=6,
            radius=140.0,
            seed=7,
        ),
    )
    rows = list(csv.DictReader((tmp_path / "mixed_edge_dataset" / "manifest.csv").open(encoding="utf-8")))
    with np.load(tmp_path / "mixed_edge_dataset" / "windows" / "mixed_window_0000000.npz") as data:
        labels = data["source_particle_id"]
        generators = result["windows"]

    assert generators == 4
    assert len(rows) == 4
    assert len(set(labels.tolist()) - {-1}) >= 2
    assert int(rows[0]["negative_edges"]) > 0


def test_model_selection_score_penalizes_split_merge_energy():
    good = {
        "ari": 0.8,
        "pairwise_f1": 0.95,
        "object_accuracy": 0.95,
        "split_rate": 0.02,
        "merge_rate": 0.02,
        "energy_error": 0.05,
    }
    bad = {**good, "split_rate": 0.4, "merge_rate": 0.3, "energy_error": 0.7}

    assert model_selection_score(good) > model_selection_score(bad)


def synthetic_window():
    return {
        "hit_x": np.array([0, 1, 2, 20, 21, 22, 80], dtype=np.float32),
        "hit_y": np.array([0, 0, 0, 20, 20, 20, 80], dtype=np.float32),
        "hit_time": np.array([0, 1, 2, 0, 1, 2, 100], dtype=np.float64),
        "hit_energy": np.ones(7, dtype=np.float32),
        "hit_tot": np.ones(7, dtype=np.float32),
        "hit_ftoa": np.zeros(7, dtype=np.float32),
        "hit_particle_id": np.array([0, 0, 0, 1, 1, 1, -1], dtype=np.int32),
    }


def write_synthetic_particle_shard(path: Path) -> None:
    window = synthetic_window()
    np.savez_compressed(
        path,
        **window,
        source_path=np.asarray("sample.t3pa"),
    )

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from particle_classification.data.edge_training import (
    EDGE_TYPE_LOCAL,
    EDGE_TYPE_MEDIUM,
    EDGE_TYPE_SAME_PIXEL,
    EDGE_TYPE_TIME,
    EdgeDatasetConfig,
    build_multiscale_edges,
    make_edge_window,
)
from particle_classification.data.edge_mixing import (
    MixedEdgeDatasetConfig,
    make_procedural_window,
)
from particle_classification.data.particles import DBSCANParticleParams
from particle_classification.models import EdgeTrackNetTiny, EdgeTrackNetTinyConfig, connected_components_from_edges


def test_normalization_json_roundtrip(tmp_path):
    particles = tmp_path / "particles"
    particles.mkdir()
    shard = particles / "sample.particles.npz"
    write_particle_shard(shard, synthetic_window())
    (particles / "manifest.csv").write_text(
        "source_path,output_path,row_count,particle_count,validation_warnings,status\n"
        f"sample.t3pa,{shard.as_posix()},7,2,,ok\n",
        encoding="utf-8",
    )
    params_path = tmp_path / "params.json"
    params_path.write_text('{"eps": 2.0, "min_samples": 2, "time_scale": 10.0}', encoding="utf-8")

    from particle_classification.data.edge_training import build_edge_training_set

    build_edge_training_set(
        particles,
        tmp_path / "dataset",
        params_path=params_path,
        config=EdgeDatasetConfig(
            window_size=10,
            window_overlap=0,
            min_hits=1,
            min_stability_ari=0.0,
            val_fraction=0.0,
            test_fraction=0.0,
        ),
    )
    normalization = json.loads((tmp_path / "dataset" / "normalization.json").read_text(encoding="utf-8"))
    row = next(csv.DictReader((tmp_path / "dataset" / "manifest.csv").open(encoding="utf-8")))
    with np.load(row["path"], allow_pickle=False) as data:
        features = data["features"]
        raw = data["raw_features"]

    assert normalization["input_dim"] == 11
    assert raw.shape == features.shape
    assert np.all(np.isfinite(features))


def test_multiscale_edges_have_expected_types():
    xyt = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [6.0, 0.0, 0.2],
            [6.0, 0.0, 0.3],
        ],
        dtype=np.float64,
    )
    x = xyt[:, 0].astype(np.float32)
    y = xyt[:, 1].astype(np.float32)

    edge_index, edge_type = build_multiscale_edges(
        xyt,
        x,
        y,
        config=EdgeDatasetConfig(k_neighbors=2, radius=2.0, medium_k_neighbors=2, medium_radius=7.0, time_neighbor_count=1),
    )

    assert edge_index.shape[1] > 0
    assert np.any((edge_type & EDGE_TYPE_LOCAL) > 0)
    assert np.any((edge_type & EDGE_TYPE_MEDIUM) > 0)
    assert np.any((edge_type & EDGE_TYPE_TIME) > 0)
    assert np.any((edge_type & EDGE_TYPE_SAME_PIXEL) > 0)


def test_bridge_pruning_prevents_single_edge_merge():
    edge_index = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    edge_scores = np.array([0.95, 0.95, 0.99], dtype=np.float32)
    object_scores = np.ones(4, dtype=np.float32)
    edge_attr = np.zeros((3, 14), dtype=np.float32)
    edge_attr[2, 3] = 8.0
    edge_attr[2, 4] = 8.0 / 128.0

    labels = connected_components_from_edges(
        4,
        edge_index,
        edge_scores,
        object_scores,
        edge_attr,
        edge_threshold=0.8,
        object_threshold=0.5,
    )

    assert labels[0] == labels[1]
    assert labels[2] != labels[3]


def test_single_hit_particle_survives_readout():
    labels = connected_components_from_edges(
        1,
        np.empty((2, 0), dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.array([0.9], dtype=np.float32),
    )

    assert labels.tolist() == [0]


def test_duplicate_pixel_hits_are_not_collapsed():
    window = synthetic_window()
    window["hit_x"] = np.array([10, 10, 10], dtype=np.float32)
    window["hit_y"] = np.array([20, 20, 20], dtype=np.float32)
    window["hit_time"] = np.array([0, 10, 20], dtype=np.float64)
    window["hit_energy"] = np.ones(3, dtype=np.float32)
    window["hit_tot"] = np.ones(3, dtype=np.float32)
    window["hit_ftoa"] = np.zeros(3, dtype=np.float32)
    window["hit_particle_id"] = np.array([0, 0, 1], dtype=np.int32)

    edge_window = make_edge_window(
        window,
        params=DBSCANParticleParams(eps=2.0, min_samples=1, time_scale=10.0),
        config=EdgeDatasetConfig(min_hits=1, min_stability_ari=0.0, same_pixel_time_radius=3.0),
    )

    assert edge_window is not None
    assert edge_window["features"].shape[0] == 3
    assert np.any((edge_window["edge_type"] & EDGE_TYPE_SAME_PIXEL) > 0)


def test_synthetic_crossing_tracks_keep_truth_labels():
    rng = __import__("random").Random(123)
    np_rng = np.random.default_rng(123)
    window, label = make_procedural_window(
        rng,
        np_rng,
        params=DBSCANParticleParams(eps=3.0, min_samples=2, time_scale=10.0),
        config=MixedEdgeDatasetConfig(min_particles=2, max_particles=2, hard_fraction=1.0),
        hard=True,
    )

    assert label == "procedural"
    assert len(set(window["hit_particle_id"].tolist()) - {-1}) == 2


def test_cpu_latency_budget_on_dense_window():
    rng = np.random.default_rng(42)
    n_hits = 160
    labels = np.repeat(np.arange(4, dtype=np.int32), n_hits // 4)
    window = {
        "hit_x": rng.normal(np.repeat([24, 70, 120, 180], n_hits // 4), 1.0).astype(np.float32),
        "hit_y": rng.normal(np.repeat([30, 75, 125, 185], n_hits // 4), 1.0).astype(np.float32),
        "hit_time": np.tile(np.arange(n_hits // 4), 4).astype(np.float64),
        "hit_energy": rng.uniform(0.1, 3.0, n_hits).astype(np.float32),
        "hit_tot": rng.uniform(1.0, 30.0, n_hits).astype(np.float32),
        "hit_ftoa": rng.uniform(0.0, 30.0, n_hits).astype(np.float32),
        "hit_particle_id": labels,
    }
    edge_window = make_edge_window(
        window,
        params=DBSCANParticleParams(eps=3.0, min_samples=2, time_scale=10.0),
        config=EdgeDatasetConfig(min_hits=1, min_stability_ari=0.0),
    )
    assert edge_window is not None

    model = EdgeTrackNetTiny(
        EdgeTrackNetTinyConfig(
            input_dim=edge_window["features"].shape[1],
            edge_attr_dim=edge_window["edge_attr"].shape[1],
            hidden_dim=32,
            edge_hidden_dim=32,
            message_passing_steps=1,
        )
    ).eval()
    features = torch.from_numpy(edge_window["features"])
    edge_index = torch.from_numpy(edge_window["edge_index"].astype(np.int64))
    edge_attr = torch.from_numpy(edge_window["edge_attr"])

    started = time.perf_counter()
    with torch.no_grad():
        output = model(features, edge_index, edge_attr)
    edge_scores = torch.sigmoid(output["edge_logits"]).numpy()
    object_scores = torch.sigmoid(output["object_logits"]).numpy()
    pred = connected_components_from_edges(
        int(features.shape[0]),
        edge_window["edge_index"],
        edge_scores,
        object_scores,
        edge_window["edge_attr"],
        edge_window["features"],
    )
    elapsed = time.perf_counter() - started

    assert pred.shape[0] == n_hits
    assert elapsed < 1.0


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


def write_particle_shard(path: Path, window: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **window, source_path=np.asarray("sample.t3pa"))

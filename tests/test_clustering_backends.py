import csv
import json
from pathlib import Path

import numpy as np

from particle_classification.clustering import dbscan_labels, dbscan_labels_pairs
from particle_classification.data.clustering_benchmark import BenchmarkConfig, benchmark_clustering_backends
from particle_classification.data.fast_clustering import (
    native_grid_dbscan_labels,
    native_stream_grid_linker_labels,
    is_backend_available,
)
from particle_classification.data.particles import (
    DBSCANParticleParams,
    adjusted_rand_index,
    cluster_hit_arrays,
    dbscan_labels_windowed,
    load_t3pa_hit_arrays,
    write_particle_npz,
)


def test_native_grid_dbscan_matches_ckdtree_pairs_on_integer_edge_cases():
    if not is_backend_available("native-grid-dbscan"):
        import pytest

        pytest.skip("native clustering extension is not built")

    features = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.2],
            [1.0, 1.0, 0.3],
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.1],
            [80.0, 80.0, 0.0],
            [80.0, 80.0, 20.0],
            [200.0, 200.0, 200.0],
        ],
        dtype=np.float64,
    )

    expected = dbscan_labels_pairs(features, eps=1.5, min_samples=2)
    observed = native_grid_dbscan_labels(
        features[:, 0],
        features[:, 1],
        features[:, 2],
        eps=1.5,
        min_samples=2,
        threads=4,
    )

    assert adjusted_rand_index(expected, observed) == 1.0
    assert np.array_equal(expected == -1, observed == -1)


def test_native_stream_grid_linker_recovers_tracks():
    if not is_backend_available("native-stream-grid-linker"):
        import pytest

        pytest.skip("native clustering extension is not built")

    x = np.array([0, 1, 2, 20, 21, 22, 10, 10, 10], dtype=np.uint16)
    y = np.array([0, 0, 0, 20, 20, 20, 10, 10, 10], dtype=np.uint16)
    t = np.array([0, 0.1, 0.2, 0, 0.1, 0.2, 0, 40, 80], dtype=np.float64)

    labels = native_stream_grid_linker_labels(x, y, t, eps=2.0, min_samples=2, threads=4)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert labels[6] == labels[7] == labels[8] == -1


def test_ckdtree_pairs_matches_neighborhood_backend_on_shapes():
    features = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.1],
            [1.0, 0.0, 0.2],
            [20.0, 20.0, 0.0],
            [20.5, 20.0, 0.1],
            [21.0, 20.0, 0.2],
            [10.0, 80.0, 10.0],
            [10.0, 80.0, 10.1],
            [100.0, 100.0, 100.0],
        ],
        dtype=np.float64,
    )

    expected = dbscan_labels(features, eps=1.25, min_samples=2)
    observed = dbscan_labels_pairs(features, eps=1.25, min_samples=2)

    assert adjusted_rand_index(expected, observed) == 1.0
    assert np.array_equal(expected == -1, observed == -1)


def test_windowed_ckdtree_pairs_matches_full_on_separated_synthetic_tracks():
    a = np.column_stack([np.zeros(20), np.zeros(20), np.arange(20) * 0.05])
    b = np.column_stack([np.ones(20) * 20, np.ones(20) * 20, np.arange(20) * 0.05])
    features = np.concatenate([a, b], axis=0)

    full = dbscan_labels_pairs(features, eps=1.2, min_samples=2)
    windowed = dbscan_labels_windowed(
        features,
        eps=1.2,
        min_samples=2,
        window_size=16,
        window_overlap=8,
        backend="ckdtree-pairs",
    )

    assert adjusted_rand_index(full, windowed) == 1.0


def test_stream_grid_linker_recovers_simple_and_time_separated_tracks():
    raw = {
        "hit_x": np.array([0, 1, 2, 20, 21, 22, 10, 10, 10], dtype=np.float32),
        "hit_y": np.array([0, 0, 0, 20, 20, 20, 10, 10, 10], dtype=np.float32),
        "hit_time": np.array([0, 1, 2, 0, 1, 2, 0, 40, 80], dtype=np.float64),
        "hit_energy": np.ones(9, dtype=np.float32),
        "hit_tot": np.ones(9, dtype=np.float32),
        "hit_ftoa": np.zeros(9, dtype=np.float32),
        "hit_particle_id": np.array([0, 0, 0, 1, 1, 1, 2, 3, 4], dtype=np.int32),
    }
    arrays = arrays_from_raw(raw, tmp_path=Path("/tmp"))
    labels, windowed = cluster_hit_arrays(
        arrays,
        DBSCANParticleParams(eps=2.0, min_samples=2, time_scale=10.0),
        backend="stream-grid-linker",
    )

    assert not windowed
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert labels[6] == labels[7] == labels[8] == -1


def test_benchmark_cli_core_writes_expected_artifacts(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_path = raw_root / "sample.t3pa"
    write_t3pa(raw_path)
    index_csv = tmp_path / "index.csv"
    index_csv.write_text("path,extension,rows,folder\nsample.t3pa,.t3pa,6,raw\n", encoding="utf-8")
    params = DBSCANParticleParams(eps=1.5, min_samples=2, time_scale=10.0)
    arrays = load_t3pa_hit_arrays(raw_path, row_count=6)
    labels, _ = cluster_hit_arrays(arrays, params)
    reference_npz = tmp_path / "reference.particles.npz"
    summary = write_particle_npz(arrays, labels, reference_npz, params=params)
    reference_manifest = tmp_path / "manifest.csv"
    with reference_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "output_path",
                "row_count",
                "particle_count",
                "noise_count",
                "noise_fraction",
                "runtime_s",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_path": "sample.t3pa",
                "output_path": reference_npz.as_posix(),
                "row_count": 6,
                "particle_count": summary["particle_count"],
                "noise_count": summary["noise_count"],
                "noise_fraction": summary["noise_fraction"],
                "runtime_s": 0.01,
                "status": "ok",
            }
        )

    result = benchmark_clustering_backends(
        raw_root,
        params,
        tmp_path / "bench",
        config=BenchmarkConfig(
            cases=("largest",),
            backends=("ckdtree-neighborhoods", "ckdtree-pairs", "stream-grid-linker", "native-grid-dbscan", "numba-grid-dbscan"),
            index_csv=index_csv.as_posix(),
            reference_manifest=reference_manifest.as_posix(),
            report_path=(tmp_path / "clustering-speed-report.md").as_posix(),
            repeat_runs=1,
            toa_tick_ns=25.0,
        ),
    )
    benchmark = list(csv.DictReader(Path(result["benchmark"]).open(encoding="utf-8")))
    agreement = list(csv.DictReader(Path(result["label_agreement"]).open(encoding="utf-8")))

    assert Path(result["benchmark"]).exists()
    assert Path(result["summary"]).exists()
    assert Path(result["report"]).exists()
    assert "rt_ratio" in benchmark[0]
    assert "physical_duration_s" in benchmark[0]
    assert any(row["backend"] == "ckdtree-pairs" and float(row["ari"]) == 1.0 for row in agreement)
    assert any(row["backend"] == "numba-grid-dbscan" and row["status"] in {"ok", "unavailable"} for row in benchmark)


def test_native_backend_npz_output_compatibility(tmp_path):
    if not is_backend_available("native-grid-dbscan"):
        import pytest

        pytest.skip("native clustering extension is not built")

    raw_path = tmp_path / "sample.t3pa"
    write_t3pa(raw_path)
    params = DBSCANParticleParams(eps=1.5, min_samples=2, time_scale=10.0)
    arrays = load_t3pa_hit_arrays(raw_path, row_count=6)
    labels, windowed = cluster_hit_arrays(arrays, params, backend="native-grid-dbscan", threads=4)
    out = tmp_path / "particles.npz"
    summary = write_particle_npz(arrays, labels, out, params=params)

    assert not windowed
    assert summary["particle_count"] == 2
    with np.load(out, allow_pickle=False) as data:
        assert data["labels_by_source_row"].shape == (6,)
        assert {"hit_x", "hit_y", "hit_time", "hit_energy", "hit_particle_id"}.issubset(data.files)


def arrays_from_raw(raw: dict[str, np.ndarray], *, tmp_path: Path):
    from particle_classification.data.particles import T3PAHitArrays

    return T3PAHitArrays(
        source_path=tmp_path / "synthetic.t3pa",
        x=raw["hit_x"].astype(np.uint16),
        y=raw["hit_y"].astype(np.uint16),
        toa=raw["hit_time"].astype(np.int64),
        time=raw["hit_time"].astype(np.float64),
        energy=raw["hit_energy"].astype(np.float32),
        tot=raw["hit_tot"].astype(np.int32),
        ftoa=raw["hit_ftoa"].astype(np.int32),
        overflow=np.zeros(raw["hit_x"].shape[0], dtype=np.int16),
        source_row=np.arange(raw["hit_x"].shape[0], dtype=np.int64),
    )


def write_t3pa(path: Path) -> None:
    rows = [
        [0, 0, 0, 4, 0, 0],
        [1, 1, 1, 4, 0, 0],
        [2, 2, 2, 4, 0, 0],
        [3, 20 + 20 * 256, 0, 4, 0, 0],
        [4, 21 + 20 * 256, 1, 4, 0, 0],
        [5, 22 + 20 * 256, 2, 4, 0, 0],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Index\tMatrix Index\tToA\tToT\tFToA\tOverflow\n")
        for row in rows:
            handle.write("\t".join(str(item) for item in row) + "\n")

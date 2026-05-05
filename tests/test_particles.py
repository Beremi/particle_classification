from pathlib import Path

import numpy as np

from particle_classification.data.particles import (
    DBSCANParticleParams,
    build_particle_outputs,
    cluster_hit_arrays,
    dbscan_labels_windowed,
    load_t3pa_hit_arrays,
    select_tuning_sample,
    write_particle_npz,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_t3pa_hit_arrays_preserves_time_energy_and_provenance():
    arrays = load_t3pa_hit_arrays(FIXTURES / "mini.t3pa")

    assert arrays.n_hits == 4
    assert arrays.x.tolist() == [0, 1, 0, 255]
    assert arrays.y.tolist() == [0, 0, 1, 1]
    assert np.allclose(arrays.time, [0.0, 9.9375, 19.875, 29.8125])
    assert np.allclose(arrays.energy, np.log1p([1, 3, 7, 11]))
    assert arrays.ftoa.tolist() == [0, 1, 2, 3]
    assert arrays.source_row.tolist() == [0, 1, 2, 3]


def test_windowed_dbscan_matches_full_dbscan_on_boundary_cluster():
    track_a = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4]], dtype=float)
    track_b = np.array([[20, 20, 0], [21, 20, 1], [22, 20, 2], [23, 20, 3], [24, 20, 4]], dtype=float)
    noise = np.array([[80, 80, 80]], dtype=float)
    features = np.vstack([track_a, track_b, noise])

    labels = dbscan_labels_windowed(features, eps=1.5, min_samples=2, window_size=6, window_overlap=3)

    assert len(set(labels.tolist()) - {-1}) == 2
    assert labels[-1] == -1
    assert len(set(labels[:5].tolist())) == 1
    assert len(set(labels[5:10].tolist())) == 1


def test_write_particle_npz_ragged_offsets_preserve_sequences(tmp_path):
    arrays = load_t3pa_hit_arrays(FIXTURES / "mini.t3pa")
    labels = np.array([0, 0, 1, -1], dtype=int)
    output = tmp_path / "mini.particles.npz"
    params = DBSCANParticleParams(eps=1.5, min_samples=2, time_scale=10.0)

    summary = write_particle_npz(arrays, labels, output, params=params)

    assert summary["particle_count"] == 2
    with np.load(output) as data:
        offsets = data["particle_offsets"]
        assert offsets.tolist() == [0, 2, 3]
        first = slice(offsets[0], offsets[1])
        second = slice(offsets[1], offsets[2])
        assert data["hit_x"][first].tolist() == [0, 1]
        assert data["hit_y"][second].tolist() == [1]
        assert data["hit_time"][first].tolist() == [0.0, 9.9375]
        assert data["hit_particle_id"].tolist() == [0, 0, 1, -1]
        assert data["labels_by_source_row"].tolist() == [0, 0, 1, -1]
        assert str(data["time_formula"]) == "relative(ToA - FToA / 16)"
        assert int(data["noise_offset"]) == 3


def test_cluster_hit_arrays_uses_x_y_time_not_energy():
    arrays = load_t3pa_hit_arrays(FIXTURES / "mini.t3pa")
    params = DBSCANParticleParams(eps=2.0, min_samples=2, time_scale=100.0)

    labels, windowed = cluster_hit_arrays(arrays, params)

    assert not windowed
    assert len(set(labels.tolist()) - {-1}) == 1


def test_cluster_hit_arrays_voxel_corner_backend_uses_continuity_rule():
    arrays = load_t3pa_hit_arrays(FIXTURES / "mini.t3pa")
    params = DBSCANParticleParams(eps=0.0, min_samples=2, time_scale=10.0)

    labels, windowed = cluster_hit_arrays(arrays, params, backend="voxel-cc-corner")

    assert not windowed
    assert labels[:3].tolist() == [0, 0, 0]
    assert labels[3] == -1


def test_select_tuning_sample_is_fixed_seed_and_stratified(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    rows = [
        ("a/small_1.t3pa", "a", 100),
        ("a/small_2.t3pa", "a", 1999),
        ("b/medium_1.t3pa", "b", 2000),
        ("b/medium_2.t3pa", "b", 10_000),
        ("b/medium_3.t3pa", "b", 20_000),
        ("b/medium_4.t3pa", "b", 49_999),
        ("c/large_1.t3pa", "c", 50_000),
        ("c/large_2.t3pa", "c", 100_000),
        ("c/large_3.t3pa", "c", 499_999),
        ("d/huge_1.t3pa", "d", 500_000),
    ]
    for path, _, _ in rows:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Index\tMatrix Index\tToA\tToT\tFToA\tOverflow\n", encoding="utf-8")
    index = tmp_path / "index.csv"
    index.write_text(
        "path,folder,extension,bytes,rows\n"
        + "\n".join(f"{path},{folder},.t3pa,1,{count}" for path, folder, count in rows)
        + "\n",
        encoding="utf-8",
    )

    sample_a = select_tuning_sample(root, index_csv=index, seed=20260502)
    sample_b = select_tuning_sample(root, index_csv=index, seed=20260502)

    assert [item.relative_path for item in sample_a] == [item.relative_path for item in sample_b]
    assert [item.tier for item in sample_a].count("small") == 2
    assert [item.tier for item in sample_a].count("medium") == 4
    assert [item.tier for item in sample_a].count("large") == 3
    assert [item.tier for item in sample_a].count("huge") == 1


def test_build_particle_outputs_smoke(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    source = root / "sample.t3pa"
    source.write_text(
        "Index\tMatrix Index\tToA\tToT\tFToA\tOverflow\n"
        "0\t0\t100\t1\t0\t0\n"
        "1\t1\t110\t2\t0\t0\n"
        "2\t2\t120\t3\t0\t0\n"
        "3\t1000\t500\t4\t0\t0\n",
        encoding="utf-8",
    )
    index = tmp_path / "index.csv"
    index.write_text("path,folder,extension,bytes,rows\nsample.t3pa,,.t3pa,1,4\n", encoding="utf-8")
    params = DBSCANParticleParams(eps=2.0, min_samples=2, time_scale=100.0)

    result = build_particle_outputs(root, tmp_path / "particles", params, index_csv=index)

    manifest = Path(result["manifest"])
    assert manifest.exists()
    output = tmp_path / "particles" / "sample.particles.npz"
    assert output.exists()
    with np.load(output) as data:
        assert data["particle_offsets"].tolist() == [0, 3]
        assert data["labels_by_source_row"].tolist() == [0, 0, 0, -1]

import csv
import json
from pathlib import Path

import numpy as np

from particle_classification.data.edge_curriculum import (
    CurriculumDatasetConfig,
    build_edge_curriculum_set,
)
from particle_classification.edge_training import EdgeWindowDataset, random_batch


def test_curriculum_manifest_normalization_and_group_splits(tmp_path):
    real_manifest = make_source_dataset(tmp_path / "real", "real", "teacher_dbscan")
    mixed_manifest = make_source_dataset(tmp_path / "mixed", "mixed", "synthetic_truth")
    normalization = tmp_path / "real" / "normalization.json"

    result = build_edge_curriculum_set(
        real_manifest=real_manifest,
        mixed_manifest=mixed_manifest,
        normalization_path=normalization,
        output_dir=tmp_path / "curriculum",
        config=CurriculumDatasetConfig(train_real_ratio=0.5, val_real_ratio=0.5, test_real_ratio=0.5),
    )

    rows = list(csv.DictReader((tmp_path / "curriculum" / "manifest.csv").open(encoding="utf-8")))
    by_group: dict[str, set[str]] = {}
    for row in rows:
        by_group.setdefault(row["split_group"], set()).add(row["split"])
    assert result["sources"] == {"real_stable": 3, "hard_mixed": 3}
    assert max(len(splits) for splits in by_group.values()) == 1

    row = next(item for item in rows if item["curriculum_source"] == "hard_mixed")
    with np.load(row["path"], allow_pickle=False) as data:
        assert np.allclose(data["features"], (data["raw_features"] - 1.0) / 2.0)
        assert str(data["curriculum_source"]) == "hard_mixed"

    copied_norm = json.loads((tmp_path / "curriculum" / "normalization.json").read_text(encoding="utf-8"))
    assert copied_norm["source"] == normalization.as_posix()
    assert copied_norm["curriculum_config"]["train_real_ratio"] == 0.5


def test_curriculum_random_batch_is_source_balanced(tmp_path):
    real_manifest = make_source_dataset(tmp_path / "real", "real", "teacher_dbscan", rows_per_split=4)
    mixed_manifest = make_source_dataset(tmp_path / "mixed", "mixed", "synthetic_truth", rows_per_split=8)
    build_edge_curriculum_set(
        real_manifest=real_manifest,
        mixed_manifest=mixed_manifest,
        normalization_path=tmp_path / "real" / "normalization.json",
        output_dir=tmp_path / "curriculum",
        config=CurriculumDatasetConfig(train_real_ratio=0.5, val_real_ratio=0.5, test_real_ratio=0.5),
    )

    dataset = EdgeWindowDataset(tmp_path / "curriculum" / "manifest.csv", split="train")
    batch = random_batch(dataset, 4)
    sources = [str(item["curriculum_source"]) for item in batch]

    assert sources.count("real_stable") == 2
    assert sources.count("hard_mixed") == 2


def make_source_dataset(
    root: Path,
    prefix: str,
    label_source: str,
    *,
    rows_per_split: int = 1,
) -> Path:
    root.mkdir(parents=True)
    rows = []
    row_id = 0
    for split in ["train", "val", "test"]:
        for local_id in range(rows_per_split):
            path = root / "windows" / split / f"{prefix}_{row_id}.npz"
            write_window(path, offset=float(row_id))
            rows.append(
                {
                    "split": split,
                    "path": path.as_posix(),
                    "source_npz": f"{prefix}_{row_id}.particles.npz",
                    "source_path": f"{prefix}_{split}_{local_id}.t3pa",
                    "split_group": f"{prefix}_{split}_{local_id}.t3pa",
                    "teacher_name": "dbscan_v001",
                    "teacher_params_json": "{}",
                    "label_source": label_source,
                    "local_window_id": local_id,
                    "n_hits": 3,
                    "n_edges": 2,
                    "positive_edges": 1,
                    "negative_edges": 1,
                    "ignored_edges": 0,
                    "particles": 2,
                    "noise_fraction": 0.0,
                    "stability_ari": 1.0,
                    "status": "ok",
                }
            )
            row_id += 1
    manifest = root / "manifest.csv"
    write_rows(manifest, rows)
    (root / "normalization.json").write_text(
        json.dumps(
            {
                "schema_version": "phase1_nodes_v2",
                "feature_names": [f"f{i}" for i in range(11)],
                "edge_attr_names": [f"e{i}" for i in range(14)],
                "input_dim": 11,
                "edge_attr_dim": 14,
                "mean": [1.0] * 11,
                "std": [2.0] * 11,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def write_window(path: Path, *, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.full((3, 11), offset + 3.0, dtype=np.float32)
    np.savez_compressed(
        path,
        features=raw.copy(),
        raw_features=raw.copy(),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_label=np.array([1, 0], dtype=np.int64),
        edge_weight=np.ones(2, dtype=np.float32),
        edge_attr=np.zeros((2, 14), dtype=np.float32),
        object_label=np.ones(3, dtype=np.float32),
        object_weight=np.ones(3, dtype=np.float32),
        source_particle_id=np.array([0, 0, 1], dtype=np.int64),
        hit_energy=np.ones(3, dtype=np.float32),
        stability_ari=np.asarray(1.0, dtype=np.float32),
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

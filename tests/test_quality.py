import csv
from pathlib import Path

import numpy as np

from particle_classification.data.quality import ContinuityAuditConfig, audit_particle_continuity, voxel_connected_components_labels


def test_continuity_audit_flags_disconnected_and_touching_labels(tmp_path):
    shard = tmp_path / "sample.particles.npz"
    np.savez(
        shard,
        hit_x=np.array([0, 1, 10, 11, 20, 21, 50, 51], dtype=np.uint16),
        hit_y=np.array([0, 0, 10, 10, 20, 21, 50, 51], dtype=np.uint16),
        hit_time=np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=np.float64),
        hit_particle_id=np.array([0, 0, 0, 0, 1, 1, 2, 3], dtype=np.int32),
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "output_path", "row_count", "status"])
        writer.writeheader()
        writer.writerow({"source_path": "sample.t3pa", "output_path": shard.as_posix(), "row_count": 8, "status": "ok"})

    result = audit_particle_continuity(
        manifest,
        tmp_path / "audit",
        config=ContinuityAuditConfig(time_bin=1.0, connectivity="corner"),
    )

    assert result["files_with_disconnected_particles"] == 1
    assert result["total_disconnected_particles"] == 1
    assert result["files_with_touching_label_pairs"] == 1
    disconnected = list(csv.DictReader(Path(result["particle_csv"]).open(encoding="utf-8")))
    assert disconnected[0]["particle_id"] == "0"
    assert disconnected[0]["component_count"] == "2"
    touching = list(csv.DictReader(Path(result["touch_csv"]).open(encoding="utf-8")))
    assert any({row["particle_a"], row["particle_b"]} == {"2", "3"} for row in touching)


def test_voxel_connected_components_preserves_corner_continuity_and_splits_gaps():
    x = np.array([0, 1, 2, 10, 11, 50], dtype=np.uint16)
    y = np.array([0, 1, 2, 10, 10, 50], dtype=np.uint16)
    t = np.array([0, 1, 2, 0, 0, 100], dtype=np.float64)

    labels = voxel_connected_components_labels(x, y, t, time_bin=1.0, connectivity="corner", min_hits=2)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert labels[5] == -1

import csv
import json
from pathlib import Path

import numpy as np

from particle_classification.data.phase2_dataset import (
    Phase2DatasetConfig,
    build_phase2_dataset,
    load_phase2_manifest,
)
from particle_classification.phase2 import Phase2ParticleDataset, collate_phase2


def test_phase2_dataset_preserves_particles_samples_large_and_groups_splits(tmp_path):
    particles = make_phase2_particle_source(tmp_path)
    result = build_phase2_dataset(
        particles,
        tmp_path / "phase2",
        config=Phase2DatasetConfig(
            max_points=5,
            large_particle_threshold=6,
            views_per_large_particle=2,
            chunk_size=3,
            val_fraction=0.0,
            test_fraction=0.0,
        ),
    )

    rows = load_phase2_manifest(tmp_path / "phase2" / "manifest.csv")
    assert result["ok_views"] == 4
    assert result["splits"]["train"] == 4
    assert max(len({row["split"] for row in rows if row["split_group"] == group}) for group in {row["split_group"] for row in rows}) == 1
    assert any(row["size_bucket"] == ">512 sampled" or row["quality_flags"] == "sampled" for row in rows)

    first = rows[0]
    with np.load(first["chunk_path"], allow_pickle=False) as data:
        start = int(data["offsets"][int(first["chunk_row"])])
        end = int(data["offsets"][int(first["chunk_row"]) + 1])
        points = data["points"][start:end]
    assert points.shape[1] == 10
    assert points.shape[0] == int(first["view_n_hits"])

    dataset = Phase2ParticleDataset(tmp_path / "phase2", split="train")
    batch = collate_phase2([dataset[0], dataset[1]])
    assert batch["points"].shape[0] == 2
    assert batch["mask"].dtype == np.bool_ or str(batch["mask"].dtype) == "torch.bool"


def make_phase2_particle_source(tmp_path: Path) -> Path:
    root = tmp_path / "particles"
    root.mkdir()
    shard = root / "toy.particles.npz"
    n = 25
    x = np.concatenate([np.array([10]), np.arange(20, 24), np.linspace(50, 70, 20)]).astype(np.uint16)
    y = np.concatenate([np.array([12]), np.arange(30, 34), np.linspace(80, 100, 20)]).astype(np.uint16)
    t = np.concatenate([np.array([0.0]), np.arange(4) * 10.0, np.arange(20) * 20.0]).astype(np.float64)
    tot = np.full(n, 20, dtype=np.int32)
    energy = np.log1p(tot).astype(np.float32)
    offsets = np.asarray([0, 1, 5, 25], dtype=np.int64)
    np.savez_compressed(
        shard,
        hit_x=x,
        hit_y=y,
        hit_time=t,
        hit_toa=t.astype(np.int64),
        hit_energy=energy,
        hit_tot=tot,
        hit_ftoa=np.zeros(n, dtype=np.int32),
        hit_overflow=np.zeros(n, dtype=np.int16),
        hit_source_row=np.arange(n, dtype=np.int64),
        hit_particle_id=np.repeat([0, 1, 2], [1, 4, 20]).astype(np.int32),
        particle_offsets=offsets,
        particle_id=np.asarray([0, 1, 2], dtype=np.int32),
        particle_n_hits=np.diff(offsets).astype(np.int32),
        particle_energy_sum=np.asarray([energy[:1].sum(), energy[1:5].sum(), energy[5:].sum()], dtype=np.float32),
        labels_by_source_row=np.repeat([0, 1, 2], [1, 4, 20]).astype(np.int32),
        source_path=np.asarray("toy.t3pa"),
        params_json=np.asarray(json.dumps({"eps": 4.0, "min_samples": 3, "time_scale": 15_000_000.0})),
    )
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["status", "output_path", "source_path", "row_count", "particle_count", "validation_warnings"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "status": "ok",
                "output_path": shard.as_posix(),
                "source_path": "toy.t3pa",
                "row_count": n,
                "particle_count": 3,
                "validation_warnings": "",
            }
        )
    return root

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ...dbscan.pipeline import write_dict_rows
from .dataset import load_manifest_rows, summarize_manifest


@dataclass(frozen=True)
class CurriculumDatasetConfig:
    seed: int = 20260503
    train_real_ratio: float = 0.50
    val_real_ratio: float = 0.50
    test_real_ratio: float = 0.50


def build_edge_curriculum_set(
    *,
    real_manifest: str | Path,
    mixed_manifest: str | Path,
    normalization_path: str | Path,
    output_dir: str | Path,
    config: CurriculumDatasetConfig | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Build a leakage-safe curriculum dataset from existing real and mixed windows.

    The output copies selected NPZ windows into a new dataset folder and applies the
    supplied normalization to preserved `raw_features`. This lets fine-tuning start
    from a checkpoint trained with the same normalization contract.
    """

    config = config or CurriculumDatasetConfig()
    output = Path(output_dir)
    windows_dir = output / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    normalization_source = Path(normalization_path)
    normalization = json.loads(normalization_source.read_text(encoding="utf-8"))

    rng = random.Random(config.seed)
    real_rows = load_manifest_rows(real_manifest)
    mixed_rows = load_manifest_rows(mixed_manifest)
    manifest_rows: list[dict[str, object]] = []

    for split, ratio in [
        ("train", config.train_real_ratio),
        ("val", config.val_real_ratio),
        ("test", config.test_real_ratio),
    ]:
        selected_real = [row for row in real_rows if row.get("split") == split]
        mixed_pool = [row for row in mixed_rows if row.get("split") == split]
        rng.shuffle(selected_real)
        rng.shuffle(mixed_pool)
        selected_mixed = select_mixed_rows_for_ratio(
            real_count=len(selected_real),
            mixed_pool=mixed_pool,
            real_ratio=ratio,
        )
        for row in selected_real:
            manifest_rows.append(
                copy_curriculum_window(
                    row,
                    source_name="real_stable",
                    output_dir=windows_dir,
                    normalization=normalization,
                    next_id=len(manifest_rows),
                )
            )
        for row in selected_mixed:
            manifest_rows.append(
                copy_curriculum_window(
                    row,
                    source_name="hard_mixed",
                    output_dir=windows_dir,
                    normalization=normalization,
                    next_id=len(manifest_rows),
                )
            )
        if verbose:
            print(
                f"{split}: real={len(selected_real)} mixed={len(selected_mixed)} "
                f"target_real_ratio={ratio:.2f}",
                flush=True,
            )

    write_dict_rows(output / "manifest.csv", manifest_rows)
    normalization_out = dict(normalization)
    normalization_out["source"] = normalization_source.as_posix()
    normalization_out["curriculum_config"] = asdict(config)
    normalization_out["curriculum_sources"] = {
        "real_manifest": Path(real_manifest).as_posix(),
        "mixed_manifest": Path(mixed_manifest).as_posix(),
    }
    (output / "normalization.json").write_text(
        json.dumps(normalization_out, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = summarize_manifest(manifest_rows)
    summary.update(
        {
            "manifest": (output / "manifest.csv").as_posix(),
            "normalization": (output / "normalization.json").as_posix(),
            "config": asdict(config),
            "sources": {
                "real_stable": sum(1 for row in manifest_rows if row.get("curriculum_source") == "real_stable"),
                "hard_mixed": sum(1 for row in manifest_rows if row.get("curriculum_source") == "hard_mixed"),
            },
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def select_mixed_rows_for_ratio(
    *,
    real_count: int,
    mixed_pool: list[dict[str, str]],
    real_ratio: float,
) -> list[dict[str, str]]:
    if real_count <= 0:
        return mixed_pool
    if real_ratio <= 0.0:
        return mixed_pool
    if real_ratio >= 1.0:
        return []
    target_mixed = int(round(real_count * (1.0 - real_ratio) / real_ratio))
    return mixed_pool[: min(len(mixed_pool), max(target_mixed, 0))]


def copy_curriculum_window(
    row: dict[str, str],
    *,
    source_name: str,
    output_dir: Path,
    normalization: dict[str, object],
    next_id: int,
) -> dict[str, object]:
    split = row.get("split", "train")
    target = output_dir / source_name / split / f"window_{next_id:07d}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with np.load(row["path"], allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    raw_features = arrays.get("raw_features", arrays["features"]).astype(np.float32)
    arrays["raw_features"] = raw_features
    arrays["features"] = normalize_features(raw_features, normalization)
    arrays["curriculum_source"] = np.asarray(source_name)
    tmp_path = target.with_name(f"{target.name}.tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(target)

    new_row: dict[str, object] = dict(row)
    new_row["path"] = target.as_posix()
    new_row["curriculum_source"] = source_name
    new_row["curriculum_source_path"] = row["path"]
    return new_row


def normalize_features(raw_features: np.ndarray, normalization: dict[str, object]) -> np.ndarray:
    mean = np.asarray(normalization.get("mean", []), dtype=np.float32)
    std = np.asarray(normalization.get("std", []), dtype=np.float32)
    if mean.shape[0] != raw_features.shape[1] or std.shape[0] != raw_features.shape[1]:
        raise ValueError(
            f"Normalization dimension mismatch: raw_features has {raw_features.shape[1]} columns, "
            f"normalization has mean={mean.shape[0]} std={std.shape[0]}."
        )
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    return ((raw_features - mean) / std).astype(np.float32)

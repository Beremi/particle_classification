from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


HUMAN_GOLD_COLUMNS = [
    "source_path",
    "window_path",
    "hit_source_row",
    "gold_particle_id",
    "gold_object_label",
    "annotator",
    "notes",
]


@dataclass(frozen=True)
class HumanGoldRow:
    source_path: str
    window_path: str
    hit_source_row: int
    gold_particle_id: int
    gold_object_label: int
    annotator: str = ""
    notes: str = ""


def load_human_gold_csv(path: str | Path) -> list[HumanGoldRow]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in HUMAN_GOLD_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Human gold CSV is missing columns: {', '.join(missing)}")
        return [
            HumanGoldRow(
                source_path=row["source_path"],
                window_path=row["window_path"],
                hit_source_row=int(row["hit_source_row"]),
                gold_particle_id=int(row["gold_particle_id"]),
                gold_object_label=int(row["gold_object_label"]),
                annotator=row.get("annotator", ""),
                notes=row.get("notes", ""),
            )
            for row in reader
        ]

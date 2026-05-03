# Human Gold Labels Schema

Phase 1 supports a future hand-corrected real validation/test set without
requiring one to exist yet. Store annotations as CSV with these columns:

| Column | Meaning |
|---|---|
| `source_path` | Original raw `.t3pa` path or acquisition identifier. |
| `window_path` | Edge-window NPZ path being annotated. |
| `hit_source_row` | Original hit row index when available, otherwise window-local hit index. |
| `gold_particle_id` | Human-corrected particle ID; use `-1` for noise. |
| `gold_object_label` | `1` for particle/object hit, `0` for noise. |
| `annotator` | Person or process that created the label. |
| `notes` | Optional free-text notes. |

Evaluation must report `human_gold` metrics separately from `teacher_dbscan`
and `synthetic_truth`.

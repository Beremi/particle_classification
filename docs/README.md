# Documentation

This directory contains topic-focused notes that complement the root [README.md](../README.md), the raw data index, and the report in [`particle_nn_report_updated/`](../particle_nn_report_updated/).

Use these documents when you need more detail on a specific part of the problem rather than the repository overview.

## Topics

- [dbscan-clustering.md](dbscan-clustering.md)
  - What DBSCAN is, what it is not, how it behaves on sparse detector hits, and when to consider related methods.

- [classification-taxonomy.md](classification-taxonomy.md)
  - Which kinds of classification are relevant for pixel detector data, from noise filtering to morphology, particle type, and frame-level analysis.

- [cluster-features-and-pipeline.md](cluster-features-and-pipeline.md)
  - Practical feature ideas for cluster-level modeling and a recommended staged pipeline for this repository.

- [raw-data-index.md](raw-data-index.md)
  - Generated inventory of the extracted raw `.t3pa` archive, with per-folder sizes, row counts, metadata notes, and largest files.

- [phase1-native-grid-baseline.md](phase1-native-grid-baseline.md)
  - Active Phase 1 separator baseline: custom C/OpenMP grid DBSCAN, full-data validation, speed results, and commands.

- [clustering-speed-report.md](clustering-speed-report.md)
  - Benchmark comparison for SciPy, Numba, native grid DBSCAN, and stream-grid linker backends.

- [phase1-nn-full-report.md](phase1-nn-full-report.md)
  - Archived neural-network separator attempt. Useful background, but not the active Phase 1 baseline.

## Relationship to the Report

The LaTeX report is the main technical narrative and literature review.

These Markdown files are meant to:

- preserve conclusions from later expert discussion and the XY-invariant design update
- make individual topics easier to find
- keep the root README readable
- provide implementation-oriented notes for the next development phase

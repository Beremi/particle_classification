# Particle Classification from Sparse Detector Matrices

This repository contains a partial project focused on classification of particle-like objects observed in sparse detector energy matrices. At the moment, the repository is primarily a research and reporting workspace: it includes one representative measurement dump, a LaTeX technical report, the compiled PDF, and the figures used in the report.

The current repository does **not** yet contain a full training or inference pipeline. Its main value is the problem definition, data interpretation, baseline recommendations, and a concrete plan for how to build a classification system on top of the available detector data.

## Repository Status

- Current form: report-first / analysis-first repository
- Main artifact: technical report in LaTeX and PDF
- Data included: one raw sparse-matrix dump
- Code included: only report build files, no ML training code yet
- Report language: Czech
- This README: English summary of the full repository

## Repository Structure

```text
.
├── README.md
├── docs/
│   ├── README.md
│   ├── classification-taxonomy.md
│   ├── cluster-features-and-pipeline.md
│   └── dbscan-clustering.md
├── matrix_dump_0001.txt
└── particle_nn_report_updated/
    ├── Makefile
    ├── README.md
    ├── particle_nn_report.tex
    ├── particle_nn_report.pdf
    ├── particle_nn_report.bib
    ├── particle_nn_report.bbl
    └── assets/
        ├── dataset_summary.png
        ├── dbscan_examples_collage.png
        ├── examples_from_dump.png
        └── temporal_structure.png
```

## What Each File Is For

- [`matrix_dump_0001.txt`](matrix_dump_0001.txt)
  - Raw measurement dump in text form.
  - Contains sparse `256x256` matrices with metadata and nonzero energy entries.
  - The report text refers to `matrix_dump_0001.zip`; this repository currently contains the extracted text dump.

- [`particle_nn_report_updated/particle_nn_report.tex`](particle_nn_report_updated/particle_nn_report.tex)
  - Main LaTeX source of the technical report.
  - Describes the data, the current DBSCAN workflow, recommended ML formulations, and relevant literature.

- [`particle_nn_report_updated/particle_nn_report.pdf`](particle_nn_report_updated/particle_nn_report.pdf)
  - Compiled report PDF for direct reading.

- [`particle_nn_report_updated/particle_nn_report.bib`](particle_nn_report_updated/particle_nn_report.bib)
  - Bibliography used by the report.

- [`particle_nn_report_updated/particle_nn_report.bbl`](particle_nn_report_updated/particle_nn_report.bbl)
  - Generated bibliography file included for easier reproducibility.

- [`particle_nn_report_updated/assets/`](particle_nn_report_updated/assets/)
  - Figures used in the report: dataset summary, temporal structure, examples from the dump, and DBSCAN screenshots.

- [`particle_nn_report_updated/Makefile`](particle_nn_report_updated/Makefile)
  - Convenience targets for building and cleaning the LaTeX document.

- [`docs/README.md`](docs/README.md)
  - Entry point for focused topic documentation derived from the report and expert discussion.

- [`docs/dbscan-clustering.md`](docs/dbscan-clustering.md)
  - Detailed explanation of DBSCAN, what it actually does on detector hits, and when to consider OPTICS, HDBSCAN, or ST-DBSCAN.

- [`docs/classification-taxonomy.md`](docs/classification-taxonomy.md)
  - Breakdown of what can be classified in this type of detector data: noise, morphology, particle type, track geometry, coincidences, frame-level states, and anomalies.

- [`docs/cluster-features-and-pipeline.md`](docs/cluster-features-and-pipeline.md)
  - Practical notes on per-cluster features, recommended pipeline structure, and risks of frame aggregation.

## Additional Documentation

The root README stays high-level. More detailed notes now live in:

- [`docs/README.md`](docs/README.md) for the documentation index
- [`docs/dbscan-clustering.md`](docs/dbscan-clustering.md) for clustering methodology
- [`docs/classification-taxonomy.md`](docs/classification-taxonomy.md) for detector-task taxonomy
- [`docs/cluster-features-and-pipeline.md`](docs/cluster-features-and-pipeline.md) for practical modeling guidance

## Project Goal

The project is centered on interpreting sparse detector measurements and building a pipeline that can eventually classify particle-related structures from them. The report makes an important distinction:

- The dump contains **measured detector hits / energy deposits**
- The desired output is a **reconstructed product**, such as:
  - clusters
  - tracks
  - object classes
  - event-level statistics

In other words, the repository is about going from raw sparse detector matrices to meaningful particle or morphology classification, not about classifying ordinary images.

## Data Overview

According to the report, the dump contains:

- `198` sparse matrices
- `40` unique `sample` blocks
- `sample` range `78` to `117`
- matrix shape always `256 x 256`
- `1` to `6` matrices per `sample`, typically `5`
- total active pixels across the dump: `187,771`
- average occupancy: about `1.45%`
- median occupancy: about `1.76%`

This is a strongly sparse dataset, which is why the report argues against treating it as a dense image problem by default.

## Record Format

Each record in the dump corresponds to one sparse matrix plus timing metadata.

### Main fields

| Field | Meaning |
|---|---|
| `sample` | Index of a temporal block in the dump |
| `set_index` | Index of a parallel matrix inside the same `sample` |
| `acq_unix` | Absolute Unix timestamp in seconds |
| `hw_t0_ns` | Hardware timing metadata in nanoseconds |
| `hw_t_proc_ns` | Additional hardware timing / processing-window metadata |
| `matrix.shape` | Always `[256, 256]` |
| `matrix.nnz` | Number of active pixels in the sparse matrix |
| `matrix.entries` | List of nonzero entries in the form `{x, y, energy}` |

### Important interpretation notes

- A pixel here is **not** a camera pixel in the usual computer-vision sense.
- Each nonzero element represents an active detector cell with calibrated energy.
- Missing coordinates are implicitly zero.
- `sample` groups matrices in time.
- Matrices within the same `sample` share the same timing metadata and differ by `set_index`.
- The exact physical meaning of `set_index`, `hw_t0_ns`, and `hw_t_proc_ns` still needs confirmation from detector or acquisition documentation.

## Key Findings from the Current Report

The report extracts several important facts from the available dump:

- Neighboring `sample` values differ by `1`.
- Neighboring `sample` blocks differ by exactly `2` seconds in `acq_unix`.
- The dump therefore spans about `78` seconds of acquisition history.
- Most frames are very sparse.
- The data morphology includes isolated deposits, short linear tracks, and compact local clusters.
- Because the data are sparse and temporally structured, simply summing frames is likely to lose useful information.

The report's main technical conclusion is that these measurements should be handled as **sparse detector hits with metadata**, not as generic dense images.

## Problem Decomposition Proposed in the Report

The report separates the full task into five layers:

1. `P1` - denoising / hit filtering
2. `P2` - clustering or instance segmentation
3. `P3` - cluster or object classification
4. `P4` - association across time or multiple views
5. `P5` - sample-level or run-level summary statistics

This is important because "particle classification" is not a single step in the current data representation. A useful system will likely need to solve at least clustering and classification, and possibly also temporal or multi-view association.

## Recommended Interpretation of the Current Objective

The report recommends treating the first practical target as:

`raw hit map -> clustering -> cluster-level classification + confidence`

That means the most realistic first deliverable is not full particle identification from raw matrices alone, but a system that:

- finds candidate clusters
- assigns each cluster a class
- reports confidence
- preserves relevant per-cluster metadata

The report explicitly recommends returning at least:

- `sample`
- `set_index`
- `cluster_id`
- cluster hit list or mask
- total energy
- size
- eccentricity
- predicted class
- confidence
- optional uncertain / out-of-distribution flag

## DBSCAN in This Project

An important terminology correction from the expert discussion:

- DBSCAN is **not** a supervised particle classifier.
- DBSCAN is an **unsupervised density-based clustering method**.
- In this repository, it should be treated as a way to group detector hits into candidate objects, not as the final particle-identification stage.

What DBSCAN actually uses depends on the feature space you give it:

- if the input is only `(x, y)`, clustering is purely spatial
- if the input is `(x, y, t)`, clustering becomes spatio-temporal
- if energy is added, for example `(x, y, log(E))`, energy can influence cluster formation

That also means DBSCAN does **not** understand energy physically by itself. It only uses energy if it is explicitly embedded into the feature space or passed through sample weights.

For the detector-style plots discussed with the expert, the practical interpretation is:

- `eps = 1.5` on a regular pixel grid with Euclidean distance behaves close to 8-neighborhood connectivity
- `min_samples = 3` suppresses isolated or very sparse hits as noise
- cluster colors are only cluster identifiers
- noise is typically represented with label `-1`

So the real role of DBSCAN here is:

- separate nearby hits into local objects
- reject isolated noise
- produce candidate clusters for later feature extraction and classification

It should **not** be expected to directly answer questions like "this was a proton" or "this was an electron" without an additional supervised layer.

## Baseline and Model Recommendations

### 1. Fastest baseline

Keep the existing DBSCAN-style clustering as the current `P2` solution, then add a cluster classifier for `P3`.

Suggested classifier inputs:

- cluster crop for a small CNN branch
- scalar features such as:
  - total energy
  - maximum energy
  - number of hits
  - shape moments
  - eccentricity
  - `set_index`
  - timing metadata

This is the recommended shortest path to a usable first system.

### 2. Best medium-term direction

Use a sparse point-set or graph formulation over active hits only.

The report treats this as the strongest practical compromise because it:

- preserves sparsity naturally
- handles variable numbers of hits
- allows time and metadata to be added directly as features
- fits multi-view or multi-frame modeling better than plain 2D CNNs

Representative model families mentioned in the report:

- GravNet / GarNet
- EdgeConv / ParticleNet-style models
- point / set transformers

### 3. Best long-term direction if ground truth exists

If hit-level or instance-level truth becomes available from simulation or high-quality annotation, the report recommends moving toward **object condensation** or an equivalent learned clustering model.

This would allow:

- instance segmentation
- clustering
- classification
- optional regression heads

in a more end-to-end way than DBSCAN.

### 4. Label-efficient second phase

If the project later grows to include a large archive of unlabeled data, the report recommends self-supervised pretraining before fine-tuning a smaller classification head.

## Why DBSCAN Is Still Kept in Scope

The report does not reject DBSCAN outright. It recommends keeping it because it is:

- interpretable
- easy to debug
- useful as a baseline
- useful for pseudo-label generation
- useful as a region-proposal stage

At the same time, the report argues that DBSCAN alone is not enough because it does not naturally solve:

- object classification
- overlapping or varying-density structures
- temporal reasoning
- multi-view association

The expert discussion adds one more practical limitation: a single global density scale can be a poor fit when the data contains both:

- long sparse tracks
- compact dense deposits

That is why related methods matter:

- `OPTICS` is useful when cluster structure exists across multiple density scales
- `HDBSCAN` is more robust to variable density
- `ST-DBSCAN` is often more natural when space, time, and additional detector features all matter

For this repository, the recommended interpretation remains:

- plain DBSCAN is a good first segmentation layer
- a separate classifier should sit on top of the resulting clusters
- more advanced clustering should be considered once variable density or temporal structure becomes the main failure mode

## Classification Tasks Beyond Clustering

For detector data of the Timepix / Medipix type, the expert discussion separates several different tasks that are easy to conflate if everything is called "particle classification".

Possible targets include:

1. signal vs. noise classification
2. morphological cluster classification
3. physical particle or interaction classification
4. track-geometry classification such as orientation, directionality, or stopping vs. through-going behavior
5. coincidence analysis across views, layers, or nearby detectors
6. frame-level or sample-level state classification
7. anomaly / pile-up / overlap detection

The same discussion also highlights two useful taxonomies:

- morphology-level classes such as `dots`, `small blobs`, `curly tracks`, `heavy blobs`, `heavy tracks`, and `straight tracks`
- physics-level classes such as `protons`, `ions`, `electrons`, `gamma/X-rays`, `thermal neutrons`, and `fast neutrons`

For the current repository state, morphology-level classification is the more realistic near-term objective. Physics-level particle identification usually needs stronger supervision, calibration, or reference datasets.

## Main Technical Guidance

The report repeatedly emphasizes the following principles:

- Preserve sparsity whenever possible.
- Do not collapse multiple frames into a simple sum unless it is only a baseline.
- Split data by `sample` or time block to avoid leakage.
- Distinguish morphological classes from true physical particle identity.
- If physics-level labels are not available, start with morphology-level classification first.
- Use DBSCAN as a "what belongs together" layer, not as the final classifier.
- Be careful with unified or aggregated frames: combining multiple frames can merge physically independent events into one cluster.

## Suggested Development Roadmap

### Stage 1

- Keep DBSCAN
- Build a cluster dataset
- Train a small cluster classifier
- Measure macro F1, purity, and merge/split behavior

### Stage 2

- Move to a point-set / GNN representation
- Include `set_index` and timing metadata in the feature set
- Compare single-frame vs. multi-frame inference

### Stage 3

- Introduce hit-level truth from simulation
- Replace or reduce dependence on DBSCAN
- Train an object-condensation-style model

### Stage 4

- Collect larger unlabeled archives
- Pretrain a sparse encoder
- Fine-tune with limited annotations

## What Is Missing from the Repository Today

This repository currently does **not** include:

- data parsing scripts
- clustering implementation
- feature extraction code
- model definitions
- training scripts
- evaluation scripts
- annotation format or labels
- environment specification for ML work

So the repository should currently be understood as a **problem statement + technical report + sample data dump**, not as a finished software package.

## Building the Report

From the report directory:

```bash
cd particle_nn_report_updated
make
```

This runs:

```bash
latexmk -pdf particle_nn_report.tex
```

Alternative manual build:

```bash
cd particle_nn_report_updated
pdflatex particle_nn_report.tex
bibtex particle_nn_report
pdflatex particle_nn_report.tex
pdflatex particle_nn_report.tex
```

### Clean commands

```bash
cd particle_nn_report_updated
make clean
make distclean
```

## Recommended Starting Point for a New Contributor

If you are starting work on this repository, the most useful order is:

1. Read [`particle_nn_report_updated/particle_nn_report.pdf`](particle_nn_report_updated/particle_nn_report.pdf)
2. Inspect [`matrix_dump_0001.txt`](matrix_dump_0001.txt) to understand the raw record structure
3. Use the report figures in [`particle_nn_report_updated/assets/`](particle_nn_report_updated/assets/) to understand sparsity and temporal layout
4. Implement a parser and cluster dataset builder
5. Reproduce the DBSCAN baseline before attempting a learned model

## Summary

This repository documents an early but well-scoped particle-classification project built around sparse detector energy matrices. Its main contribution today is a technical roadmap:

- understand the dump as sparse detector-hit data
- keep DBSCAN as a baseline, not as the final answer
- first build cluster-level classification
- then move toward sparse point-set / GNN models
- use learned clustering only once suitable ground truth is available

For the current stage of the project, the report is the primary source of truth and the best guide for how the software part of the repository should be built next.

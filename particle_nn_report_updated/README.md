# LaTeX Project: XY-Invariant Particle NN Report

This folder contains the updated English technical report for the repository. The report treats the `xy_invariant_particle_report_project.zip` guidance as the primary design contract:

- candidate inputs are variable-size `(x, y, t, e)` point sets;
- detector-plane azimuth `theta_xy` is computed and stored as metadata;
- `theta_xy` is excluded from class embeddings and clustering distance;
- the first recommended model is deterministic XY-invariant descriptors plus a compact PointNet/DeepSets baseline.

## Contents

- `particle_nn_report.tex` - main LaTeX document
- `particle_nn_report.bib` - bibliography
- `particle_nn_report.pdf` - compiled PDF
- `assets/` - figures retained from the earlier data-orientation report

## Build

```bash
latexmk -pdf particle_nn_report.tex
```

or manually:

```bash
pdflatex particle_nn_report.tex
bibtex particle_nn_report
pdflatex particle_nn_report.tex
pdflatex particle_nn_report.tex
```

# Particle classification from sparse Timepix hits

Research code for turning Timepix T3PA hit streams into candidate particles
and exploring compact, XY-invariant morphology representations. The maintained
pipeline is:

```text
T3PA tables + sidecars -> validated hit arrays -> native grid DBSCAN
                       -> particle NPZ shards -> exploratory autoencoders
```

This is a research testbed, not a validated particle-species classifier. The
active Phase 1 separator is the custom C/OpenMP `native-grid-dbscan` backend.
Neural separator attempts and detailed experiment reports are retained for
reproducibility, but they are not the production path.

## Current status

- The canonical DBSCAN configuration is frozen in
  [`dbscan_phase1_native_eps5_v001.json`](configs/teachers/dbscan_phase1_native_eps5_v001.json):
  `eps=5`, `min_samples=2`, `time_scale=0.625`, 250,000-hit windows, and
  25,000-hit overlap.
- Large raw data, derived shards, caches, checkpoints, and releases stay under
  gitignored `local_data/`; Git contains parsers, a small matrix example, and a
  snapshot of the indexed original corpus.
- The repository layout before this cleanup remains browsable at the
  [`precleanup` tag](https://github.com/Beremi/particle_classification/tree/precleanup).
- `particle-fetch-raw` downloads the pinned public data release, verifies its
  SHA-256 and internal manifest, and restores the expected `local_data/raw`
  layout. `particle-raw-archive` provides the underlying deterministic pack,
  verify, and safe-unpack workflow.
- Autoencoder code is explicitly experimental. The practical reference is a
  centered `8 x 32 x 32` voxel model with an 8D latent; a pose-separated path
  model is the research direction for explicit XY invariance.

## Repository map

```text
configs/                         frozen and historical experiment parameters
data/                            tracked small example and raw-data index CSV
docs/                            compact current documentation
experimental_notes/              archived reports, figures, and background
notebooks/                       interactive demonstrations
scripts/
  data/                          raw-data and archive utilities
  dbscan/                        Phase 1 diagnostics and visualizations
  autoencoders/                  Phase 2 research workflows
  legacy/                        archived neural-separator report generators
src/particle_classification/
  data/                          T3PA parsing, metadata, indexing, archives
  dbscan/                        maintained custom DBSCAN implementation
  experiments/baseline/         small XY-invariant PointNet/DeepSets baseline
  experiments/autoencoders/      point, path, and voxel models
  experiments/neural_separator/  non-production learned separator research
  commands/                      installed command-line entry points
tests/                           parser, clustering, archive, and workflow tests
local_data/                      untracked inputs and generated artifacts
```

## Install and check

Python 3.11 or newer, a C compiler, and NumPy headers are required. An editable
install builds the native extension; Linux enables OpenMP by default.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Set `PARTICLE_DISABLE_OPENMP=1` during installation for a serial native build.
Optional dependency groups are `clustering`, `phase2`, `viz`, `parquet`, and
`speed`. Install `.[dev,phase2,viz]` when working with the archived analysis
and plotting scripts rather than only the maintained data/DBSCAN path.

## Quick start: raw T3PA to particle shards

After installing the project, download and verify the published
[raw-data v1 release](https://github.com/Beremi/particle_classification/releases/tag/raw-data-v1):

```bash
particle-fetch-raw
```

On a clean checkout this creates the gitignored working layout:

```text
local_data/
├── raw/              original 711-file T3PA corpus used by current scripts
├── alpha/t3pa/       eight newer alpha T3PA acquisitions
├── alpha/clog/       newer frame-mode alpha CLOG acquisitions
├── metadata/         delivered calibration/configuration/readme files
├── MANIFEST.json
├── SHA256SUMS
└── SOURCE_LAYOUT.json
```

The command downloads the
[release archive](https://github.com/Beremi/particle_classification/releases/download/raw-data-v1/particle-raw-v1.tar.xz),
currently 335,294,368 bytes, checks its SHA-256 against the value pinned in
this package, verifies every payload hash, and only then creates `local_data`.
It refuses an existing `local_data` tree so local experiments cannot be
overwritten accidentally.

To use the original supplied ZIP instead, place it at `raw_data.zip` and run:

```bash
particle-extract-raw raw_data.zip --dest local_data/raw
```

Build one compressed particle shard per indexed T3PA table with the active
separator:

```bash
particle-build-particles \
  --input local_data/raw \
  --index data/raw_data_index.csv \
  --params configs/teachers/dbscan_phase1_native_eps5_v001.json \
  --out local_data/processed/particles_aligned_time_eps5_v001 \
  --backend native-grid-dbscan \
  --threads 0
```

`--threads 0` lets the backend choose its default. See the
[DBSCAN guide](docs/dbscan.md) for timing coordinates, window reconciliation,
output arrays, and validation requirements.

## Pack and restore raw data

The release-to-analysis path is deliberately explicit:

```text
GitHub Release -> pinned HTTPS download -> archive SHA-256 + manifest check
               -> safe unpack -> local_data/raw -> DBSCAN particle building
```

For a manual download with GitHub CLI, fetch both release assets, check the
external checksum, verify the internal manifest, and unpack into the repository
root:

```bash
gh release download raw-data-v1 \
  --repo Beremi/particle_classification \
  --pattern 'particle-raw-v1.tar.xz*'
sha256sum --check particle-raw-v1.tar.xz.sha256
particle-raw-archive verify particle-raw-v1.tar.xz
particle-raw-archive unpack particle-raw-v1.tar.xz .
particle-raw-archive verify-tree local_data
```

Both unpacking paths refuse an existing `local_data` root unless overwrite is
explicit. `--overwrite` replaces that entire tree, including derived work, so
it should normally be used only in an empty checkout.

The publishing workflow stores the deduplicated extracted tree as deterministic
`tar.xz` with `MANIFEST.json` and `SHA256SUMS`. It verifies every payload hash
and rejects links, special files, and unsafe archive members.

On the 1.58 GB deduplicated combined corpus, XZ preset 6 produced a 335.45 MB
archive using about 97 MiB pack memory. The literal-smallest 7z result was only
4.92 MB smaller while using about 2.64 GiB, so preset 6 remains the documented
default; see the [full comparison](docs/raw-data.md#measured-lossless-compression).

Maintainers reproduce the combined source tree with
`scripts/data/stage_raw_release.py`, then pack it with archive root
`local_data`; the exact commands and source mapping are in the
[raw-data guide](docs/raw-data.md#published-release-and-reproduction). Do not
wrap the duplicate source ZIPs inside another archive.

## Documentation

- [Documentation index](docs/README.md)
- [Raw data, T3PA/CLOG formats, and publishing](docs/raw-data.md)
- [Tracked raw-data index snapshot](docs/raw-data-index.md)
- [Native grid DBSCAN](docs/dbscan.md)
- [Particle autoencoders](docs/autoencoder.md)
- [Script guide](scripts/README.md)
- [Experimental archive](experimental_notes/README.md)
- [Background LaTeX report](experimental_notes/background/particle_nn_report_updated/particle_nn_report.tex)
- [Demo notebook](notebooks/demo_data.ipynb)

## Scientific limitations

- The repository and delivered data currently have no explicit license file.
  Public download availability does not by itself grant reuse rights; a data
  license and citation should be added once the owner confirms them.
- DBSCAN produces candidate partitions, not human-verified particles or
  particle-species labels. Close events can merge and tracks can split.
- `ToT` is currently transformed into a feature proxy, not a calibrated energy.
  Detector/run calibration bindings and parts of the acquisition provenance
  are incomplete, especially for the newer E03 alpha data.
- Nonzero T3PA overflow records are filtered from particle building, but the
  corresponding lost intervals cannot be reconstructed.
- CLOG is frame-integrated data without hit-level arrival times and is not
  compatible with the T3PA/DBSCAN pipeline.
- Centered voxel inputs are not inherently rotation invariant. Detector-plane
  angle must remain metadata rather than a particle-class feature; the
  pose-separated path approach is still experimental.
- Autoencoder neighborhoods and unsupervised clusters do not establish physical
  classes without independent labels and external experimental ground truth.

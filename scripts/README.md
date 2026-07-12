# Repository Scripts

Run scripts from the repository root with the project environment active, for
example:

```bash
.venv/bin/python scripts/dbscan/render_long_timespan_particles.py --help
```

The subdirectories separate stable data and DBSCAN utilities from research
workflows:

- `data/` contains raw-data inspection, indexing, dataset preparation, the
  public-release downloader (`fetch_raw.py`), deterministic archive wrapper
  (`raw_archive.py`), and deduplicated release stager
  (`stage_raw_release.py`).
- `dbscan/` contains Phase 1 separator diagnostics, visualizations, and gallery
  generators for the canonical DBSCAN workflow.
- `autoencoders/` contains exploratory Phase 2 point, path, pose-separated, and
  voxel autoencoder training and analysis scripts.
- `legacy/` contains archived neural-separator report generators retained for
  reproducibility; these are not the production Phase 1 path.

Most commands use repository-relative input and output paths. Invoke them from
the repository root unless a script explicitly documents otherwise.

The DBSCAN and autoencoder visualization/analysis scripts need the optional
packages installed by `python -m pip install -e ".[dev,phase2,viz]"`.

Prefer the installed `particle-fetch-raw` command for a clean checkout. The
installed `particle-raw-archive` command handles pack/verify/unpack;
`data/fetch_raw.py` and `data/raw_archive.py` expose the same interfaces when
running directly from a source checkout. `data/stage_raw_release.py` is the
maintainer-only step that deduplicates the delivered source tree before
packing.

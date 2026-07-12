# Raw data

Large detector data are intentionally outside Git. The repository tracks a
small matrix example, a generated index, parsers, and processing code; local
archives and extracted corpora live below `local_data/`.

## Data collections

Keep these collections distinct:

1. `data/matrix_dump_0001.txt` is a small tracked sparse-matrix example. It
   contains `256 x 256` matrices with `(x, y, energy)` entries plus `sample`,
   `set_index`, `acq_unix`, `hw_t0_ns`, and `hw_t_proc_ns`. It is not T3PA hit
   data, and the hardware timestamp fields are not documented as SPIDR time.
2. `local_data/raw/` is the indexed T3PA working corpus extracted from the
   original archive. The tracked [index snapshot](raw-data-index.md) records
   711 `.t3pa` files, 711 `.t3pa.info` sidecars, and 44,725,206 table rows in
   six run folders. Of those rows, 75 are data-loss markers rather than hits.
3. `local_data/imports/raw data ruzna for skuta 2026/` is a later source/import
   tree. Its ZIP files include source archives for several indexed
   groups, configuration and calibration files, plus alpha-source T3PA and
   CLOG acquisitions. It is not represented completely by the tracked index.

Four imported T3PA ZIPs are exact duplicates of subsets already present in the
original corpus when members are compared by uncompressed size and CRC. Only
the two alpha-source ZIPs add measurements. A release still needs cryptographic
checksums and a manifest that explicitly marks duplicates, replacements, and
genuinely new acquisitions.

## What each collection represents

| collection | file pairs / duration | detector metadata | measurement context known from delivery | important missing context |
|---|---:|---|---|---|
| `08_thu_proton_daily_batch` | 368 × 20 s | G07-W0050, 80 V | proton-therapy-centre QA daily batch | beam energy, flux/dose, geometry, run purpose/labels |
| `13_tue_proton_daily_batch` | 229 × 20 s | G07-W0050, 80 V | second proton QA day | same gaps as above |
| `data I05` | 3 × 60 s | I05-W0044, 300 V | microtron electrons, synchronized two-detector stack, folder says 75° | beam energy/current, exact geometry/orientation and synchronization contract |
| `data M07` | 3 × 60 s | M07-W0044, 100 V | same synchronized electron acquisition | same gaps as above |
| `D05` | 54 × 20 s | D05-W0105, 80 V | proton-therapy-centre acquisition, delivery implies 500 µm Si top detector | a supplied calibration folder says I05 rather than D05; beam and geometry details missing |
| `F08` | 54 × 20 s | F08-W0060, −500 V | paired proton acquisition, delivery implies 2,000 µm CdTe bottom detector | beam and geometry details missing |
| new alpha T3PA | 8 × 20 s | E03-W0096, 80 V | table-top alpha source, “various geometries” | isotope, activity/energy, source age, distances/angles and their run-number mapping; E03 calibration/configuration |
| new alpha CLOG | 1,235 × 1 ms | E03-W0096, 80 V | same alpha session, frame-mode iToT + Count | relationship to the eight T3PA runs, third-channel calibration state, geometry/source metadata; E03 calibration/configuration |

The `.info` sidecars make the device and acquisition settings reasonably
self-describing. They do **not** encode enough experimental intent to assign a
physical label or reconstruct the source/beam geometry for each run.

## Structural validation

The original archive is file-complete relative to its own convention: 711 data
files, 711 matching sidecars, no orphans, exact headers, zero malformed rows,
and a sequential `Index` in every file. Matrix indices span the expected
`0..65535`. Its sidecars consistently contain acquisition series/time,
chipboard ID, 19 DAC values, HV, interface, detector type, Pixet version,
shutter/start times, and threshold.

The new alpha T3PA ZIP is equally well formed: eight data/sidecar pairs,
3,314,336 rows in total, no malformed rows, no overflow records, and per-run
row counts of 698, 637, 70,979, 342,384, 142,481, 1,074,856, 1,682,285, and
16 for runs 0 through 7. The extreme count variation is real in the files, but
cannot be interpreted without the missing run-to-geometry mapping. These
sidecars use the same metadata schema plus `Repeat count`.

## Expected contents of each raw file

The structural contract is known; there is no expected fixed number of hits or
clusters per acquisition.

| file | required contents | status in the delivery |
|---|---|---|
| original or alpha `*.t3pa` | one tab-separated header followed by zero or more six-integer records: `Index`, `Matrix Index`, `ToA`, `ToT`, `FToA`, `Overflow` | all 719 delivered files conform |
| `*.t3pa.info` | sidecar sections for acquisition, device/DAC settings, HV/interface, detector/Pixet version, shutter/start time, and threshold | every T3PA file has one; alpha sidecars also include `Repeat count` |
| alpha `*.clog` | one frame header followed by zero or more cluster lines containing `[x, y, iToT-or-energy, count]` tuples | all 1,235 files are readable; empty frames are valid |
| alpha `*.clog.idx` | little-endian unsigned 64-bit byte offsets to frames in the matching CLOG file | each delivered one-frame file contains exactly one zero offset |
| alpha `*.clog.info` | acquisition/device sidecar for the matching CLOG frame | every CLOG file has one |

What is incomplete is not the byte/column layout. It is the scientific meaning
needed around those records: run labels and geometry, beam/source conditions,
calibration binding, timing/synchronization contract, and publication
provenance. The CLOG third tuple value is the one field-level ambiguity: the
mode permits raw iToT or calibrated energy, and the producer has not confirmed
which was exported.

## T3PA hit table

Here, “raw” means the earliest delivered input to this repository. T3PA is a
Pixet-exported ASCII pixel stream with device-extended timing, not the lowest
level TPX3 communication dump (`.t3r`). No `.t3r` data were delivered.

The supported `.t3pa` format is a tab-separated table with six integer fields:

| field | repository interpretation |
|---|---|
| `Index` | Measurement-line index. It normally grows from zero, but resets if another measurement is appended to the same file. Preserved as `hit_source_row`. |
| `Matrix Index` | Pixel address on a `256 x 256` sensor: `x = index % 256`, `y = index // 256`. ADVACAM describes `(0, 0)` as the physical lower-left MiniPIX pixel; plotting orientation must therefore be chosen explicitly. |
| `ToA` | Coarse time of arrival in 25 ns ticks. |
| `ToT` | Raw time-over-threshold value. Current models use it only as an uncalibrated energy/charge proxy. |
| `FToA` | Exported 5-bit fine timing correction (`0..31`) used in the timestamp equation below. |
| `Overflow` | Data-loss or other special-record indicator, not a normal pixel-value field. |

Fine time is reconstructed as

```text
time_ticks = ToA - FToA / 16
time_ns = 25 * time_ticks
```

Phase 1 subtracts the minimum fine time in each file before clustering. The
files do not establish the timing epoch, rollover/reset behavior, relation to
an absolute/SPIDR clock, or cross-detector synchronization accuracy.

### Overflow markers

ADVACAM defines `Overflow = 1` with matrix index `0x74` (116) as the start of
lost data and `0x75` (117) as its end; the latter row's ToA is the missing-time
length. Matrix index zero denotes a corruption event. `Overflow = 10` can hold
an external-trigger timestamp when that firmware feature is enabled.

The original corpus contains 75 such records, all with `Overflow = 1`: 37
starts and 38 ends, split between D05 (26) and F08 (49). They are not physics
hits; F08 run 28 begins with the unmatched extra end marker. The streaming
parser preserves them losslessly. The derived particle loader excludes
nonzero-overflow records by default, records how many it
skipped, and offers an explicit diagnostic option to include them. The missing
intervals themselves are unrecoverable and should be reported in analyses.

### Sidecars

Each indexed T3PA file is paired with `.t3pa.info`. Sidecars provide typed
metadata including acquisition duration in seconds, chipboard ID, high voltage
in volts, readout interface, detector type, Pixet version, start/shutter times,
and threshold in keV. The parser is generic and does not enforce a versioned
sidecar schema.

See the [T3PA parser](../src/particle_classification/data/t3pa.py), [sidecar
parser](../src/particle_classification/data/info.py), and [indexer](../src/particle_classification/data/index.py).

## CLOG is a separate format

The alpha import includes 1,235 `.clog`, `.clog.idx`, and `.clog.info`
triplets. These are 1 ms frame-mode **iToT + Count** acquisitions. Each text
file contains

```text
Frame <number> (<Unix start>, <acquisition seconds> s)
[x, y, iToT-or-calibrated-energy, event-count] ...
```

Each nonempty line after the header is one cluster. The fourth tuple value is
event count, not per-hit ToA; this frame mode contains no additional ToA. The
binary `.clog.idx` is an array of little-endian 64-bit byte offsets to frame
records. Because every delivered file contains one frame starting at byte
zero, all delivered index members contain one zero offset. `.clog.info`
provides the acquisition metadata.

The delivered set has 612 frames with pixels and 623 empty frames, 1,589
cluster lines, and 39,371 pixel tuples. Tuple ranges are x/y `0..255`, iToT
`1..245`, and count `1..3`. Whether the third value is raw iToT or already
calibrated energy is not stated explicitly; its integer values and file name
strongly indicate raw iToT, but that must be confirmed by the producer before
calling it energy.

CLOG is incompatible with the T3PA pipeline: it is frame-integrated and has no
hit-level arrival time. Do not rename it to `.t3pa`, concatenate it with hit
tables, or feed it to Phase 1. Supporting it requires a separate, tested parser
and a task-specific frame/cluster workflow.

Format references: [ADVACAM file types](https://wiki.advacam.cz/wiki/File_types)
and [TraX Engine CLOG modes](https://wiki.advacam.cz/wiki/TraX_Engine_CLI).

## Calibration and provenance gaps

Current hit processing uses

```text
hit_energy = log1p(ToT)
```

and some voxel models divide that value by `log1p(1023)`. This is feature
scaling, not energy calibration. ADVACAM documents the role of per-pixel `a`,
`b`, `c`, and `t` coefficients, but this delivery still lacks a reliable
binding from every run to a calibration/configuration version, its validity
interval and uncertainty, and a documented mask/bad-pixel policy. Some folder
names also disagree on D05 versus I05 detector identity, and no E03 calibration
or configuration accompanies the alpha data, so filename inference is unsafe.

A publishable manifest should bind every archive member to:

- detector/chipboard, sensor material and thickness, geometry, and calibration;
- source or beam species, energy, flux/dose, facility, and acquisition purpose;
- timing epoch/reset/synchronization and trigger configuration;
- acquisition duration, software/config versions, exclusions, and checksums;
- license, creator, collection date, and citation/provenance information.

## Measured lossless compression

The publishing candidates were measured on a deduplicated extracted corpus,
not on a sample: 5,169 files and 1,577,930,849 logical bytes comprising the
original corpus once, both genuinely new alpha collections, and all 26 loose
configuration/calibration/readme files. ZIP containers, including the four
duplicate measurement ZIPs, were excluded. Every result was integrity-tested,
fully extracted, and matched against all source SHA-256 hashes and byte counts.

| format and settings | archive bytes | source retained | pack wall time | peak pack RSS |
|---|---:|---:|---:|---:|
| tar + 7z LZMA2, 256 MiB dictionary, level 9 | 330,533,981 | 20.95% | 1,207 s | 2,768,852 KiB |
| tar.xz, preset 9 extreme, 4 threads | 331,800,252 | 21.03% | 1,148 s | 3,761,396 KiB |
| **tar.xz, preset 6, 1 thread** | **335,450,384** | **21.26%** | **905 s** | **99,408 KiB** |
| tar.zst, level 19, long mode, 1 thread | 375,288,975 | 23.78% | 1,420 s | 531,880 KiB |

Archive sizes are exact. Pack times are indicative only: the later candidates
overlapped each other and unrelated high-CPU work on this host, so this was not
an isolated throughput benchmark. The corpus-content manifest SHA-256 is
`167b907852e78bc982b4182946a5b5b70f26e8a3b5543968ad25539648b67fa2`.

The literal minimum was 7z, but it saved only 4,916,403 bytes over XZ preset 6
while requiring roughly 28 times the pack memory and a less convenient release
toolchain. XZ preset 9 extreme saved only 3,650,132 bytes and used roughly 38
times the memory. The recommended public default is therefore deterministic
`tar.xz` at preset 6. Use `--preset 9 --extreme` only when a few megabytes are
worth substantially higher resource use. High-level Zstandard decoded fastest
in this measurement, but its archive was 39,838,591 bytes larger than XZ preset
6; lower Zstandard levels remain useful when packaging speed matters more than
release size.

## Published release and reproduction

The deduplicated corpus is published with the repository as
[raw-data v1](https://github.com/Beremi/particle_classification/releases/tag/raw-data-v1).
On a clean checkout, the supported installation path is:

```bash
particle-fetch-raw
```

`particle-fetch-raw` downloads
[`particle-raw-v1.tar.xz`](https://github.com/Beremi/particle_classification/releases/download/raw-data-v1/particle-raw-v1.tar.xz),
requires the archive to match the SHA-256 pinned in the installed code, checks
the self-verifying archive, and safely restores this layout:

```text
local_data/
├── raw/              # original corpus once; current script default
├── alpha/
│   ├── t3pa/         # extracted new alpha T3PA payload
│   └── clog/         # extracted new alpha CLOG payload
├── metadata/         # all 26 supplied non-ZIP ancillary files
├── MANIFEST.json     # generated payload inventory
├── SHA256SUMS        # generated per-file hashes
└── SOURCE_LAYOUT.json
```

The archive contains 5,169 measurement/metadata payload files totaling
1,577,930,849 bytes before the small generated inventories. Its
`SOURCE_LAYOUT.json` records component byte counts, component tree digests,
both alpha source-ZIP hashes, and the six excluded ZIP paths. The two alpha
ZIPs are represented by their extracted payloads; the four other ZIPs duplicate
data already under `raw/`.

The published archive is 335,294,368 bytes. Its SHA-256 is
`6d1696e77b3d925367239bef11d7ce1929d169a85676d3700606e0d39eeb1831`;
the same value is available as the release asset
[`particle-raw-v1.tar.xz.sha256`](https://github.com/Beremi/particle_classification/releases/download/raw-data-v1/particle-raw-v1.tar.xz.sha256).

For a manual installation, download both assets and run every verification
step explicitly:

```bash
gh release download raw-data-v1 \
  --repo Beremi/particle_classification \
  --pattern 'particle-raw-v1.tar.xz*'
sha256sum --check particle-raw-v1.tar.xz.sha256
particle-raw-archive verify particle-raw-v1.tar.xz
particle-raw-archive unpack particle-raw-v1.tar.xz .
particle-raw-archive verify-tree local_data
```

Both paths refuse to replace an existing `local_data` tree by default.
`--overwrite` replaces that entire root, including derived experiments, and is
therefore intended only for a disposable or empty checkout.

### Maintainer reproduction trace

The exact release flow is source delivery → safe/deduplicated staging →
deterministic pack → verification → GitHub Release:

```bash
IMPORTS='local_data/imports/raw data ruzna for skuta 2026'
ALPHA="$IMPORTS/TPX3 alpha zaric mereni na stole PIXET 17feb2026"

.venv/bin/python scripts/data/stage_raw_release.py \
  /tmp/particle-raw-v1-stage \
  --original-raw local_data/raw \
  --imports-root "$IMPORTS" \
  --alpha-t3pa-zip "$ALPHA/01 data s pixetem zaric.zip" \
  --alpha-clog-zip "$ALPHA/03 clog 1ms config z aug2025 zaric.zip"

particle-raw-archive pack \
  /tmp/particle-raw-v1-stage \
  /tmp/particle-raw-v1.tar.xz \
  --root local_data \
  --preset 6
particle-raw-archive verify /tmp/particle-raw-v1.tar.xz
(cd /tmp && sha256sum particle-raw-v1.tar.xz \
  > particle-raw-v1.tar.xz.sha256)
```

The stager and packer reject links, special files, unsafe archive members, and
existing output trees unless overwrite is explicit. For `raw-data-v1`, the
stager also requires exact checked-in component counts, byte totals, tree
digests, both alpha source-ZIP hashes, and the six excluded ZIP paths. The
escape hatch `--allow-unrecognized-delivery` is for synthetic tests or a future
named release and must not be used to rebuild v1. The packer preserves every
regular payload byte, sorts members, normalizes archive metadata, adds
`MANIFEST.json` and `SHA256SUMS`, and verifies the completed output. Identical
input bytes, root name, preset, Python/liblzma, and code version produce the
same archive.

The original source ZIP can still populate only the historical working corpus:

```bash
particle-extract-raw raw_data.zip --dest local_data/raw
```

That extractor likewise preflights paths and member types and extracts through
a temporary directory. The generated hashes prove file identity, but they do
not replace the incomplete scientific provenance and unspecified data license
described above.

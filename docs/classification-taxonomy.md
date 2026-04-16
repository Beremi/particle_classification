# Classification Taxonomy for Pixel Detector Data

## Why This Taxonomy Matters

In this repository, "particle classification" can mean several different tasks. Mixing them together makes the project harder to define and harder to evaluate.

The main distinction is:

- clustering asks **which hits belong together**
- classification asks **what the resulting object means**

The current repository is much closer to the first problem than to full physics-level particle identification.

## Levels of Classification

For this type of sparse detector data, at least seven different targets are relevant.

### 1. Signal vs. Noise

Decide whether a hit or cluster is:

- useful detector signal
- noise
- artifact

This is often the first filtering step before any cluster-level interpretation.

### 2. Morphological Cluster Classification

Classify the shape or appearance of a cluster.

This is usually the most realistic first supervised task when full physics labels are missing.

Example morphology classes mentioned in the expert discussion:

- `dots`
- `small blobs`
- `curly tracks`
- `heavy blobs`
- `heavy tracks`
- `straight tracks`

These are useful because morphology often already contains information about the interaction pattern, even before full physical interpretation is possible.

### 3. Physical Particle or Interaction Classification

Map a cluster or track to a physical hypothesis such as:

- `protons`
- `ions`
- `electrons`
- `gamma/X-rays`
- `thermal neutrons`
- `fast neutrons`

This is a stronger objective than morphology classification and usually requires:

- calibration
- reference datasets
- simulation truth
- or carefully labeled experimental data

### 4. Track Geometry Classification

Classify geometric behavior such as:

- orientation
- directionality
- stopping vs. through-going behavior
- curvature or straightness

This can be useful even when physics identity is still uncertain.

### 5. Coincidence Classification

Determine whether structures across:

- multiple layers
- multiple views
- neighboring detectors
- nearby time windows

belong to the same event or interaction.

### 6. Frame-Level or Sample-Level State Classification

Instead of classifying a single cluster, classify the condition of a whole frame or sample block, for example:

- background-dominated frame
- beam burst
- mixed field
- detector state change

### 7. Anomaly / Pile-Up / Overlap Detection

Identify situations such as:

- overlapping tracks
- pile-up
- saturation
- abnormal detector behavior
- out-of-distribution patterns

## What Is Most Realistic for This Repository Right Now

Based on the current contents of the repository:

- there is one sparse dump
- there is no explicit label set
- there is no annotation format
- there is no simulation truth in the repository

So the most realistic near-term supervised target is:

- morphology-level classification of DBSCAN or otherwise reconstructed clusters

The less realistic near-term target is:

- full physics-level particle identification from raw 2D matrices alone

## Recommended Target Order

For this repository, a pragmatic order is:

1. signal vs. noise
2. cluster morphology
3. geometric track attributes
4. physics-level particle or interaction class
5. multi-view / time association
6. frame-level summaries and anomaly detection

This sequence matches the amount of supervision and detector context typically required.

## Implication for Evaluation

Different targets require different metrics.

Examples:

- signal vs. noise: precision / recall
- morphology: macro F1 and confusion matrix
- clustering + morphology: purity, efficiency, merge rate, split rate
- physics PID: per-class efficiency vs. energy and detector position
- anomalies: false positive rate and calibration of uncertainty

## Sources

- Timepix review article discussed by the expert
  - <https://www.mdpi.com/2410-390X/8/1/17>
- Data Processing Engine (DPE) paper referenced in the expert discussion
  - <https://arxiv.org/abs/2310.15723>

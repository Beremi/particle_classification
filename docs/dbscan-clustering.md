# DBSCAN and Related Clustering Methods

## Summary

For this repository, DBSCAN should be understood as a **clustering / segmentation tool**, not as a supervised particle classifier.

Its practical role is:

- take sparse detector hits
- group nearby hits into candidate objects
- reject isolated points as noise
- hand the resulting clusters to a later classification stage

## What DBSCAN Is

DBSCAN belongs to the family of **density-based clustering** methods.

At a high level it:

- marks points as core points when they have enough neighbors inside radius `eps`
- grows clusters from connected regions of core points
- attaches border points to those clusters
- labels isolated points as noise

Important consequence:

- DBSCAN does not require the number of clusters in advance
- DBSCAN can recover clusters with irregular shapes
- DBSCAN is unsupervised

## What DBSCAN Is Not

DBSCAN is not a supervised classifier in the sense of:

- learning particle classes from labeled training examples
- directly predicting `proton`, `electron`, `gamma`, and so on

If a DBSCAN plot shows different colors, those colors are just **cluster IDs**, not physical classes. Noise is commonly assigned label `-1`.

## What DBSCAN Actually Uses to Decide

DBSCAN only sees:

- a set of feature vectors
- a distance metric
- density parameters such as `eps` and `min_samples`

That means the result depends entirely on the feature space you provide.

Examples:

- input `(x, y)` means purely spatial clustering
- input `(x, y, t)` means spatio-temporal clustering
- input `(x, y, log(E))` means energy can influence grouping

DBSCAN does not understand detector physics on its own. Energy, time, and detector metadata only affect the result if they are explicitly encoded into the feature space or used through sample weighting.

## Interpreting the Current Detector Use Case

For a regular pixel grid with Euclidean distance:

- `eps = 1.5` behaves close to 8-neighborhood connectivity
- horizontal, vertical, and diagonal neighbors are likely to connect
- more distant hits stay separated

With:

- `min_samples = 3`

the method tends to:

- suppress isolated pixels
- suppress very sparse local groups
- preserve denser local structures

So in this project DBSCAN is best viewed as:

- hit segmentation into local islands
- a denoising aid
- a region-proposal stage for cluster-level modeling

## Strengths for This Repository

DBSCAN remains useful here because it is:

- simple
- interpretable
- easy to tune and debug
- capable of handling irregular cluster shapes
- usable without knowing the number of clusters in advance

It is therefore a good first baseline for turning sparse hit maps into candidate objects.

## Limitations for Detector Data

Plain DBSCAN has several structural limitations in this context:

- one global density scale may not fit both sparse tracks and dense deposits
- overlapping or merged structures are hard to separate
- time is ignored unless explicitly included
- energy is ignored unless explicitly included
- clustering quality directly limits any later classifier built on top of it

In practical terms, DBSCAN answers:

- which hits appear to belong together locally

It does not answer:

- what particle or interaction created the cluster

## Related Methods

When plain DBSCAN starts failing, the following variants are relevant:

### OPTICS

Useful when the data has meaningful cluster structure across multiple density scales. Instead of committing to one `eps`, it reveals how clustering changes across scales.

### HDBSCAN

A better choice when the data contains clusters with very different local densities. This is often relevant when sparse tracks and compact deposits appear in the same detector frames.

### ST-DBSCAN

A natural extension when clustering should depend on:

- space
- time
- optionally additional non-spatial features

For detector data with pixel position, timing, and energy context, this can be more realistic than pure 2D clustering.

## Practical Recommendation for This Repository

Use DBSCAN as:

- a first-pass segmentation method
- an interpretable baseline
- a source of cluster candidates and pseudo-labels

Do not use it as:

- the final particle classifier
- the only method if variable density or temporal structure becomes dominant

Recommended progression:

1. Run DBSCAN on hits to obtain candidate clusters.
2. Compute per-cluster features.
3. Train a supervised classifier on those cluster features.
4. Move to HDBSCAN, ST-DBSCAN, or learned clustering once the baseline is understood.

## Sources

The notes in this file were condensed from the report plus the expert discussion supplied with the repository update request.

- Ester et al., "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise"
  - <https://cdn.aaai.org/KDD/1996/KDD96-037.pdf>
- scikit-learn DBSCAN documentation
  - <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>

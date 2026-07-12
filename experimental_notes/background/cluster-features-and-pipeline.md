# Cluster Features and Recommended Pipeline

## Recommended Pipeline

For the current repository state, the most practical staged workflow is:

1. read sparse detector hits from the dump
2. optionally filter obvious noise
3. cluster hits into candidate objects
4. compute per-cluster features
5. run a supervised classifier on cluster features
6. optionally aggregate cluster predictions into frame-level or run-level summaries

This keeps clustering and classification as separate steps, which matches both the report and the expert discussion.

## Why This Separation Helps

It makes the project easier to build in stages:

- clustering can be debugged visually
- cluster features can be inspected directly
- classification labels can be introduced later
- future learned clustering can replace the clustering stage without rewriting the whole pipeline

## Candidate Per-Cluster Features

The expert discussion suggests treating DBSCAN output as a candidate-object representation and then classifying clusters using features such as:

- total energy
- maximum pixel energy
- number of active hits
- bounding-box size
- area
- major-axis length
- minor-axis length
- eccentricity
- compactness
- orientation
- curvature
- time span
- `set_index`
- `hw_t0_ns`
- `hw_t_proc_ns`

These can be combined with:

- a crop-based image branch
- a point-set branch over hits within the cluster
- or a pure tabular classifier

## Recommended Near-Term Model Shapes

Practical options for the first supervised layer:

### Small CNN + Scalar Features

Good when:

- clusters can be cropped into small windows
- morphology matters strongly
- implementation speed matters more than full generality

### Point-Set / Graph Classifier

Good when:

- cluster size varies a lot
- sparsity should be preserved
- energy and time features should remain explicit

This aligns with the report's medium-term recommendation.

## Caution on Unified or Aggregated Frames

The expert discussion highlights an important risk:

- if multiple frames are merged into one unified frame, DBSCAN may connect physically independent events into the same cluster

That changes the task from:

- particle-by-particle clustering and classification

to something closer to:

- classification of a time-integrated pattern

So unified frames should be treated carefully:

- acceptable as a baseline visualization
- risky as the main representation for particle-level classification

Whenever possible, preserve:

- frame identity
- time ordering
- per-sample grouping

## When Plain DBSCAN Stops Being Enough

Move beyond plain DBSCAN if you start seeing:

- clusters with very different densities in the same frame
- frequent merge or split failures
- strong timing dependence
- too much sensitivity to one global `eps`

At that point, reasonable next steps are:

- HDBSCAN for variable density
- ST-DBSCAN for spatio-temporal clustering
- point-set / GNN models for cluster classification
- object-condensation-style models once good truth labels exist

## Practical Recommendation for This Repository

The most defensible first implementation is:

`raw hits -> DBSCAN -> cluster features -> supervised cluster classifier`

That gives a usable baseline while keeping the door open for:

- better clustering
- better labels
- learned end-to-end reconstruction later

## Sources

- Root repository overview
  - [../README.md](../../README.md)
- Technical report
  - [particle_nn_report_updated/particle_nn_report.pdf](particle_nn_report_updated/particle_nn_report.pdf)
- DPE paper referenced in the expert discussion
  - <https://arxiv.org/abs/2310.15723>

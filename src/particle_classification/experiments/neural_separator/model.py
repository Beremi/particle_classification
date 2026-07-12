from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

@dataclass(frozen=True)
class EdgeTrackNetTinyConfig:
    input_dim: int = 7
    edge_attr_dim: int = 3
    hidden_dim: int = 64
    edge_hidden_dim: int = 64
    dropout: float = 0.05
    message_passing_steps: int = 0


class EdgeTrackNetTiny(nn.Module):
    """Hit-level edge-link model for DBSCAN replacement experiments.

    The model predicts local same-particle links and per-hit objectness. A
    connected-components readout over high-confidence edges gives variable-size
    particles without choosing a fixed number of output instances.
    """

    def __init__(self, config: EdgeTrackNetTinyConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = EdgeTrackNetTinyConfig(**kwargs)
        self.config = config
        h = config.hidden_dim
        edge_h = config.edge_hidden_dim
        edge_input_dim = h * 3 + config.edge_attr_dim
        self.hit_mlp = nn.Sequential(
            nn.Linear(config.input_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )
        self.message_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        for _ in range(config.message_passing_steps):
            self.message_mlps.append(
                nn.Sequential(
                    nn.Linear(edge_input_dim, h),
                    nn.LayerNorm(h),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(h, h),
                )
            )
            self.update_mlps.append(
                nn.Sequential(
                    nn.Linear(h * 2, h),
                    nn.LayerNorm(h),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(h, h),
                )
            )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_h),
            nn.LayerNorm(edge_h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(edge_h, edge_h // 2),
            nn.SiLU(),
            nn.Linear(edge_h // 2, 1),
        )
        self.object_head = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.SiLU(),
            nn.Linear(h // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError("features must have shape [nodes, channels].")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges].")
        h = self.hit_mlp(features)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        aux = prepare_edge_attr(features, src, dst, edge_attr, self.config.edge_attr_dim) if src.numel() else None
        if src.numel():
            for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps, strict=True):
                h_src = h[src]
                h_dst = h[dst]
                messages = message_mlp(torch.cat([h_src, h_dst, h_src - h_dst, aux], dim=-1))
                aggregate = torch.zeros_like(h)
                aggregate.index_add_(0, dst, messages)
                counts = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
                counts.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
                aggregate = aggregate / torch.clamp(counts, min=1.0)
                h = h + update_mlp(torch.cat([h, aggregate], dim=-1))
        if src.numel():
            h_src = h[src]
            h_dst = h[dst]
            delta = h_src - h_dst
            edge_logits = self.edge_mlp(torch.cat([h_src, h_dst, delta, aux], dim=-1)).squeeze(-1)
        else:
            edge_logits = torch.empty(0, dtype=features.dtype, device=features.device)
        return {
            "edge_logits": edge_logits,
            "object_logits": self.object_head(h).squeeze(-1),
            "node_embedding": h,
        }


def edge_aux_features(features: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    xyt = features[:, :3]
    energy = features[:, 3]
    delta_xyt = xyt[src] - xyt[dst]
    distance = torch.linalg.norm(delta_xyt, dim=-1, keepdim=True)
    delta_t = torch.abs(features[src, 2:3] - features[dst, 2:3])
    delta_e = torch.abs(energy[src] - energy[dst]).unsqueeze(-1)
    return torch.cat([distance, delta_t, delta_e], dim=-1)


def prepare_edge_attr(
    features: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_attr: torch.Tensor | None,
    edge_attr_dim: int,
) -> torch.Tensor:
    if edge_attr is None:
        aux = edge_aux_features(features, src, dst)
    else:
        aux = edge_attr.to(device=features.device, dtype=features.dtype)
    if aux.shape[1] == edge_attr_dim:
        return aux
    if aux.shape[1] > edge_attr_dim:
        return aux[:, :edge_attr_dim]
    padding = torch.zeros((aux.shape[0], edge_attr_dim - aux.shape[1]), dtype=aux.dtype, device=aux.device)
    return torch.cat([aux, padding], dim=-1)


def connected_components_from_edges(
    n_nodes: int,
    edge_index,
    edge_scores,
    object_scores=None,
    edge_attr=None,
    node_features=None,
    *,
    edge_threshold: float = 0.5,
    object_threshold: float = 0.5,
    bridge_pruning: bool = True,
):
    """Return particle IDs from edge/object scores; `-1` means noise."""

    import numpy as np

    edge_index = np.asarray(edge_index)
    edge_scores = np.asarray(edge_scores)
    edge_attr = None if edge_attr is None else np.asarray(edge_attr)
    node_features = None if node_features is None else np.asarray(node_features)
    if object_scores is None:
        active = np.ones(n_nodes, dtype=bool)
    else:
        active = np.asarray(object_scores) >= object_threshold
    parent = np.arange(n_nodes, dtype=np.int64)
    rank = np.zeros(n_nodes, dtype=np.int8)

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = int(parent[idx])
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        if rank[root_a] == rank[root_b]:
            rank[root_a] += 1

    score_lookup = {(int(edge_index[0, idx]), int(edge_index[1, idx])): float(edge_scores[idx]) for idx in range(edge_index.shape[1])}
    for idx in range(edge_index.shape[1]):
        score = float(edge_scores[idx])
        if not bridge_edge_passes(
            idx,
            edge_index,
            edge_attr,
            score,
            score_lookup,
            edge_threshold=edge_threshold,
            bridge_pruning=bridge_pruning,
        ):
            continue
        a = int(edge_index[0, idx])
        b = int(edge_index[1, idx])
        if active[a] and active[b]:
            union(a, b)

    root_to_label: dict[int, int] = {}
    labels = np.full(n_nodes, -1, dtype=np.int32)
    for node_idx in range(n_nodes):
        if not active[node_idx]:
            continue
        root = find(node_idx)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[node_idx] = root_to_label[root]
    if bridge_pruning and node_features is not None:
        labels = split_suspicious_components(labels, node_features)
    return labels


def bridge_edge_passes(
    idx: int,
    edge_index,
    edge_attr,
    score: float,
    score_lookup: dict[tuple[int, int], float],
    *,
    edge_threshold: float,
    bridge_pruning: bool,
) -> bool:
    a = int(edge_index[0, idx])
    b = int(edge_index[1, idx])
    reverse = score_lookup.get((b, a))
    if reverse is None:
        if score < edge_threshold + (0.05 if bridge_pruning else 0.0):
            return False
    elif (score + reverse) * 0.5 < edge_threshold:
        return False
    if bridge_pruning and edge_attr is not None and edge_attr.shape[1] >= 6:
        dist_xy = float(edge_attr[idx, 4])
        abs_dt = float(edge_attr[idx, 3])
        if dist_xy > (7.0 / 128.0) and abs_dt > 7.0:
            return False
    return True


def split_suspicious_components(labels, node_features):
    import numpy as np

    out = labels.copy()
    next_label = int(np.max(out)) + 1 if np.any(out >= 0) else 0
    for label in sorted(set(int(item) for item in out.tolist()) - {-1}):
        nodes = np.flatnonzero(out == label)
        if nodes.size < 12:
            continue
        coords = np.asarray(node_features[nodes, :3], dtype=np.float64)
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        cov = centered.T @ centered / max(nodes.size - 1, 1)
        eig = np.linalg.eigvalsh(cov)
        total = float(np.sum(np.maximum(eig, 0.0)))
        linearity = float(np.max(eig) / total) if total > 1e-12 else 1.0
        order = np.argsort(coords[:, 2], kind="mergesort")
        sorted_time = coords[order, 2]
        gaps = np.diff(sorted_time)
        if gaps.size == 0:
            continue
        max_gap_idx = int(np.argmax(gaps))
        median_gap = float(np.median(gaps)) if gaps.size else 0.0
        if linearity < 0.35 and float(gaps[max_gap_idx]) > max(0.15, 2.0 * median_gap):
            right_nodes = nodes[order[max_gap_idx + 1 :]]
            if right_nodes.size:
                out[right_nodes] = next_label
                next_label += 1
    return relabel_components(out)


def relabel_components(labels):
    import numpy as np

    out = np.full_like(labels, -1)
    mapping: dict[int, int] = {}
    for idx, label in enumerate(labels.tolist()):
        label = int(label)
        if label < 0:
            continue
        if label not in mapping:
            mapping[label] = len(mapping)
        out[idx] = mapping[label]
    return out

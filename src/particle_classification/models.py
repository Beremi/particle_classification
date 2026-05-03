from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class XYInvariantParticleNetConfig:
    input_dim: int = 4
    hidden_dim: int = 64
    class_dim: int = 48
    dropout: float = 0.05


class XYInvariantParticleNet(nn.Module):
    """Compact PointNet/DeepSets baseline.

    The model consumes canonicalized point sets. Its `z_class` output is the
    only vector intended for clustering/classification; pose metadata heads are
    reported separately and must not be concatenated into class distance.
    """

    def __init__(self, config: XYInvariantParticleNetConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = XYInvariantParticleNetConfig(**kwargs)
        self.config = config
        h = config.hidden_dim
        self.hit_mlp = nn.Sequential(
            nn.Linear(config.input_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )
        self.attention = nn.Linear(h, 1)
        self.class_head = nn.Sequential(
            nn.Linear(h * 3, h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.class_dim),
        )
        self.pose_head = nn.Sequential(nn.Linear(h * 3, h), nn.SiLU(), nn.Linear(h, 2))
        self.q_head = nn.Sequential(nn.Linear(h * 3, h // 2), nn.SiLU(), nn.Linear(h // 2, 1), nn.Sigmoid())
        self.tilt_head = nn.Sequential(nn.Linear(h * 3, h // 2), nn.SiLU(), nn.Linear(h // 2, 2))
        self.profile_head = nn.Sequential(nn.Linear(h * 3, h), nn.SiLU(), nn.Linear(h, 24))

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, object]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, points, channels].")
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        else:
            mask = mask.to(device=features.device, dtype=torch.bool)

        h = self.hit_mlp(features)
        pooled = self._pool(h, mask)
        z_class = self.class_head(pooled)

        theta_unit = torch.nn.functional.normalize(self.pose_head(pooled), dim=-1)
        tilt_unit = torch.nn.functional.normalize(self.tilt_head(pooled), dim=-1)
        theta_xy = torch.atan2(theta_unit[:, 1], theta_unit[:, 0])
        phi_t = torch.atan2(tilt_unit[:, 1], tilt_unit[:, 0])

        return {
            "z_class": z_class,
            "metadata": {
                "theta_xy": theta_xy,
                "theta_xy_unit": theta_unit,
                "q_theta": self.q_head(pooled).squeeze(-1),
                "phi_t": phi_t,
                "phi_t_unit": tilt_unit,
            },
            "profile": self.profile_head(pooled),
        }

    def _pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=h.dtype)
        denominator = torch.clamp(mask_f.sum(dim=1), min=1.0)
        mean_pool = (h * mask_f).sum(dim=1) / denominator

        max_input = h.masked_fill(~mask.unsqueeze(-1), torch.finfo(h.dtype).min)
        max_pool = max_input.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        att_logits = self.attention(h).squeeze(-1).masked_fill(~mask, torch.finfo(h.dtype).min)
        att_weights = torch.softmax(att_logits, dim=1).unsqueeze(-1)
        att_pool = (h * att_weights).sum(dim=1)
        return torch.cat([max_pool, mean_pool, att_pool], dim=-1)


@dataclass(frozen=True)
class EdgeTrackNetTinyConfig:
    input_dim: int = 7
    hidden_dim: int = 64
    edge_hidden_dim: int = 64
    dropout: float = 0.05


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
        self.hit_mlp = nn.Sequential(
            nn.Linear(config.input_dim, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.SiLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(h * 3 + 3, edge_h),
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

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError("features must have shape [nodes, channels].")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges].")
        h = self.hit_mlp(features)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        if src.numel():
            h_src = h[src]
            h_dst = h[dst]
            delta = h_src - h_dst
            aux = edge_aux_features(features, src, dst)
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


def connected_components_from_edges(
    n_nodes: int,
    edge_index,
    edge_scores,
    object_scores=None,
    *,
    edge_threshold: float = 0.5,
    object_threshold: float = 0.5,
):
    """Return particle IDs from edge/object scores; `-1` means noise."""

    import numpy as np

    edge_index = np.asarray(edge_index)
    edge_scores = np.asarray(edge_scores)
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

    for idx in range(edge_index.shape[1]):
        if edge_scores[idx] < edge_threshold:
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
    return labels

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

    The model consumes canonicalized point sets. Its ``z_class`` output is the
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

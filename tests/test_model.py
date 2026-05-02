import numpy as np
import torch

from particle_classification.geometry import ParticleCandidate, rotate_xy
from particle_classification.models import XYInvariantParticleNet
from particle_classification.training import collate_candidates, xy_invariance_loss


def test_model_shapes_and_invariance_loss_are_finite():
    candidate = ParticleCandidate(
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.2, 2.0],
                [2.0, 0.0, 0.4, 3.0],
            ]
        )
    )
    batch = collate_candidates([candidate, rotate_xy(candidate, np.pi / 5)])
    model = XYInvariantParticleNet(input_dim=4, hidden_dim=16, class_dim=8, dropout=0.0)
    out = model(batch["features"], batch["mask"])
    assert out["z_class"].shape == (2, 8)
    assert out["metadata"]["theta_xy"].shape == (2,)
    loss = xy_invariance_loss(out["z_class"][0:1], out["z_class"][1:2])
    assert torch.isfinite(loss)

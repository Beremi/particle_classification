import math

import numpy as np
import torch

from particle_classification.phase2_path_ae import (
    CanonicalPathAutoencoder,
    PathAEConfig,
    canonical_energy_path,
    energy_path_tensor,
    path_ae_loss,
)


def make_line(angle=0.0, n=64):
    u = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    c, s = math.cos(angle), math.sin(angle)
    x = u * c
    y = u * s
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    e = (0.2 + 0.8 * np.exp(-u * u)).astype(np.float32)
    return np.column_stack([x, y, t, t, e, np.zeros((n, 5), dtype=np.float32)])


def test_canonical_energy_path_separates_xy_rotation():
    base, base_meta = canonical_energy_path(make_line(angle=0.2), path_points=64)
    rotated, rot_meta = canonical_energy_path(make_line(angle=1.1), path_points=64)
    assert base.shape == (64, 4)
    assert rotated.shape == (64, 4)
    np.testing.assert_allclose(base[:, :4], rotated[:, :4], atol=1e-4)
    delta = math.atan2(math.sin(rot_meta["theta_xy"] - base_meta["theta_xy"]), math.cos(rot_meta["theta_xy"] - base_meta["theta_xy"]))
    assert abs(abs(delta) - 0.9) < 1e-3


def test_path_autoencoder_forward_and_loss_are_finite():
    path, _ = canonical_energy_path(make_line(), path_points=32)
    batch = torch.from_numpy(np.stack([path, path], axis=0))
    model = CanonicalPathAutoencoder(PathAEConfig(latent_dim=8, hidden_dim=32, path_points=32, fourier_frequencies=3))
    out = model(batch)
    assert out["z"].shape == (2, 8)
    assert out["reconstruction"].shape == (2, 32, 4)
    loss, metrics = path_ae_loss(out["reconstruction"], batch)
    assert torch.isfinite(loss)
    assert np.isfinite(metrics["energy_path_relative_l2"])
    assert energy_path_tensor(batch).shape == (2, 32, 4)

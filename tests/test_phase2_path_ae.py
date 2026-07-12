import math

import numpy as np
import torch

from particle_classification.phase2_path_ae import (
    CanonicalPathAutoencoder,
    PathAEConfig,
    PoseSeparatedPathAutoencoder,
    StructuredTransformAutoencoder,
    TransformAEConfig,
    centered_transform_path,
    canonical_energy_path,
    energy_path_tensor,
    path_ae_loss,
    pose_path_ae_loss,
    pose_separated_energy_paths,
    rotate_path_by_pose,
    structured_transform_ae_loss,
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


def test_pose_separated_paths_keep_rotation_as_pose():
    base_can, base_target, base_pose, _ = pose_separated_energy_paths(make_line(angle=0.2), path_points=64)
    rot_can, rot_target, rot_pose, _ = pose_separated_energy_paths(make_line(angle=1.1), path_points=64)
    assert base_can.shape == (64, 4)
    assert base_target.shape == (64, 4)
    np.testing.assert_allclose(base_can, rot_can, atol=1e-4)
    assert not np.allclose(base_target[:, :2], rot_target[:, :2])
    delta = math.atan2(
        float(rot_pose[1] * base_pose[0] - rot_pose[0] * base_pose[1]),
        float(rot_pose[0] * base_pose[0] + rot_pose[1] * base_pose[1]),
    )
    assert abs(abs(delta) - 0.9) < 1e-3
    recovered = rotate_path_by_pose(torch.from_numpy(base_can[None]), torch.from_numpy(base_pose[None])).numpy()[0]
    np.testing.assert_allclose(recovered, base_target, atol=1e-5)


def test_pose_separated_autoencoder_forward_and_loss_are_finite():
    canonical, target, pose, _ = pose_separated_energy_paths(make_line(angle=0.7), path_points=32)
    canonical_batch = torch.from_numpy(np.stack([canonical, canonical], axis=0))
    target_batch = torch.from_numpy(np.stack([target, target], axis=0))
    pose_batch = torch.from_numpy(np.stack([pose, pose], axis=0))
    for architecture in ["implicit", "conv"]:
        model = PoseSeparatedPathAutoencoder(
            PathAEConfig(latent_dim=8, hidden_dim=32, path_points=32, fourier_frequencies=3, architecture=architecture)
        )
        out = model(canonical_batch, pose_batch)
        assert out["z_shape"].shape == (2, 8)
        assert out["canonical_reconstruction"].shape == (2, 32, 4)
        assert out["reconstruction"].shape == (2, 32, 4)
        loss, metrics = pose_path_ae_loss(out, target_batch, canonical_batch)
        assert torch.isfinite(loss)
        assert np.isfinite(metrics["frame_energy_path_relative_l2"])


def test_structured_transform_autoencoder_forward_and_loss_are_finite():
    path, metadata = centered_transform_path(make_line(angle=0.4), path_points=32)
    assert path.shape == (32, 4)
    assert abs(float(path[:, :3].mean())) < 1e-6
    assert metadata["n_hits"] == 64
    batch = torch.from_numpy(np.stack([path, path], axis=0))
    model = StructuredTransformAutoencoder(TransformAEConfig(shape_latent_dim=8, hidden_dim=64, path_points=32))
    out = model(batch)
    assert out["z_shape"].shape == (2, 8)
    assert out["raw_transform"].shape == (2, 7)
    assert out["reconstruction"].shape == (2, 32, 4)
    loss, metrics = structured_transform_ae_loss(out, batch)
    assert torch.isfinite(loss)
    assert np.isfinite(metrics["energy_path_relative_l2"])

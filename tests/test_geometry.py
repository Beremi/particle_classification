import numpy as np

from particle_classification.features import xy_invariant_descriptor
from particle_classification.geometry import ParticleCandidate, angle_distance_mod_pi, rotate_xy, weighted_pca_geometry


def _line_candidate(angle=0.0):
    x = np.linspace(-5, 5, 11)
    y = np.zeros_like(x)
    c = np.cos(angle)
    s = np.sin(angle)
    xy = np.column_stack([c * x - s * y + 30.0, s * x + c * y + 40.0])
    t = np.linspace(0, 1, x.size)
    e = np.linspace(1, 2, x.size)
    return ParticleCandidate(np.column_stack([xy, t, e]), candidate_id="line")


def test_weighted_geometry_tracks_rotation_as_metadata():
    candidate = _line_candidate(0.0)
    rotated = rotate_xy(candidate, np.pi / 3.0)
    g0 = weighted_pca_geometry(candidate)
    g1 = weighted_pca_geometry(rotated)
    assert angle_distance_mod_pi(g1.theta_xy - g0.theta_xy, np.pi / 3.0) < 1e-6
    assert g0.q_theta > 0.9


def test_descriptor_excludes_theta_xy_and_stays_stable_under_rotation():
    candidate = _line_candidate(0.0)
    rotated = rotate_xy(candidate, np.pi / 4.0)
    d0 = xy_invariant_descriptor(candidate)
    d1 = xy_invariant_descriptor(rotated)
    assert "theta_xy" not in d0.names
    np.testing.assert_allclose(d0.vector, d1.vector, atol=1e-6)
    assert abs(d0.metadata["theta_xy"] - d1.metadata["theta_xy"]) > 0.1

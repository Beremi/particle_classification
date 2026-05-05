import numpy as np
import torch

from particle_classification.phase2 import (
    Phase2ModelConfig,
    Phase2ParticleModel,
    collate_phase2,
    phase2_loss,
)


def test_phase2_backbones_accept_variable_size_batches():
    batch = make_batch()
    for backbone in ["deepsets", "edgeconv", "settransformer", "pointtransformer"]:
        model = Phase2ParticleModel(
            Phase2ModelConfig(
                backbone=backbone,
                objective="ae",
                point_dim=10,
                summary_dim=22,
                hidden_dim=32,
                latent_dim=16,
                decoder_points=8,
                edgeconv_k=3,
            )
        )
        out = model(batch["points"], batch["mask"], batch["summary"])
        assert out["z"].shape == (4, 16)
        assert out["decoded"].shape == (4, 8, 10)
        assert torch.isfinite(out["z"]).all()


def test_phase2_query_decoder_accepts_variable_size_batches():
    batch = make_batch()
    model = Phase2ParticleModel(
        Phase2ModelConfig(
            backbone="deepsets",
            objective="ae",
            point_dim=10,
            summary_dim=22,
            hidden_dim=32,
            latent_dim=8,
            decoder_points=12,
            decoder_arch="query",
        )
    )
    out = model(batch["points"], batch["mask"], batch["summary"])
    assert out["z"].shape == (4, 8)
    assert out["decoded"].shape == (4, 12, 10)
    assert torch.isfinite(out["decoded"]).all()


def test_phase2_objective_losses_are_finite():
    batch = make_batch()
    for objective in ["ae", "denoising_ae", "masked_ae", "contrastive", "dec", "vade"]:
        model = Phase2ParticleModel(
            Phase2ModelConfig(
                backbone="deepsets",
                objective=objective,
                point_dim=10,
                summary_dim=22,
                hidden_dim=32,
                latent_dim=16,
                decoder_points=8,
                n_clusters=4,
            )
        )
        loss, metrics = phase2_loss(model, batch, objective=objective)
        assert torch.isfinite(loss)
        assert np.isfinite(metrics["loss"])


def make_batch():
    items = []
    rng = np.random.default_rng(123)
    for idx, n_points in enumerate([1, 3, 7, 11]):
        items.append(
            {
                "points": rng.normal(size=(n_points, 10)).astype(np.float32),
                "summary": rng.normal(size=(22,)).astype(np.float32),
                "row": {"source_path": f"toy_{idx}.t3pa", "particle_id": idx},
                "n_hits": n_points,
                "view_n_hits": n_points,
                "size_bucket": "1-3" if n_points <= 3 else "4-10",
            }
        )
    return collate_phase2(items)

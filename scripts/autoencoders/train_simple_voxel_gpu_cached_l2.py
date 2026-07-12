from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from particle_classification.dbscan.pipeline import write_dict_rows
from particle_classification.experiments.autoencoders.canonical_voxel import _render_rows
from particle_classification.experiments.autoencoders.voxel import (
    VoxelGridConfig,
    VoxelMLPConfig,
    VoxelPatchMLPAutoencoder,
    VoxelSparseCache,
    corrupt_voxel_batch,
)


def load_split_to_gpu(
    cache: Path,
    split: str,
    grid: VoxelGridConfig,
    *,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict[str, str]]]:
    dataset = VoxelSparseCache(cache, split=split, grid_config=grid, seed=seed)
    rows = dataset.rows
    batches: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        dense = dataset.rows_to_dense(rows[start : start + batch_size], device=device).to(dtype=dtype)
        batches.append(dense)
        if start and start % (batch_size * 25) == 0:
            print(f"loaded {split} {start:,}/{len(rows):,}", flush=True)
    tensor = (
        torch.cat(batches, dim=0)
        if batches
        else torch.empty((0, grid.t_bins, grid.y_bins, grid.x_bins), device=device, dtype=dtype)
    )
    print(
        f"loaded {split}: {tuple(tensor.shape)} in {time.perf_counter() - t0:.1f}s, "
        f"{tensor.numel() * tensor.element_size() / 1e9:.2f} GB",
        flush=True,
    )
    return tensor, rows


def corrupt_fixed(
    target: torch.Tensor,
    seed: int,
    *,
    blur_kernel: int = 5,
    blur_mix: float = 1.0,
    noise_std: float = 0.04,
    voxel_dropout: float = 0.08,
) -> torch.Tensor:
    torch.manual_seed(seed)
    if target.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return corrupt_voxel_batch(
        target,
        blur_kernel=blur_kernel,
        blur_mix=blur_mix,
        noise_std=noise_std,
        voxel_dropout=voxel_dropout,
    )


def per_particle_plain_l2(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = reconstruction.float() - target.float()
    return torch.sqrt(torch.sum(diff * diff, dim=(1, 2, 3)))


def per_particle_relative_l2(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_norm = torch.sqrt(torch.sum(target.float() * target.float(), dim=(1, 2, 3))).clamp_min(1e-8)
    return per_particle_plain_l2(reconstruction, target) / target_norm


def plain_tensor_l2_loss(output: dict[str, torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = output["reconstruction"]
    per_particle = per_particle_plain_l2(reconstruction, target)
    relative = per_particle_relative_l2(reconstruction, target)
    target_f = target.float()
    recon_f = reconstruction.float()
    target_energy = target_f.sum(dim=(1, 2, 3)).clamp_min(1e-8)
    recon_energy = recon_f.sum(dim=(1, 2, 3))
    energy_relative = torch.mean(torch.abs(recon_energy - target_energy) / target_energy)
    overlap = torch.mean(torch.sum(torch.minimum(recon_f, target_f), dim=(1, 2, 3)) / target_energy)
    loss = per_particle.mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "plain_tensor_l2": float(loss.detach().cpu()),
        "plain_tensor_l2_p50": float(torch.quantile(per_particle.detach(), 0.50).cpu()),
        "plain_tensor_l2_p95": float(torch.quantile(per_particle.detach(), 0.95).cpu()),
        "relative_tensor_l2": float(relative.mean().detach().cpu()),
        "relative_tensor_l2_p95": float(torch.quantile(relative.detach(), 0.95).cpu()),
        "energy_relative_l1": float(energy_relative.detach().cpu()),
        "mass_overlap": float(overlap.detach().cpu()),
    }


def scan_losses(
    model: VoxelPatchMLPAutoencoder,
    data: torch.Tensor,
    *,
    batch_size: int,
    seed: int,
    corrupted: bool,
    blur_kernel: int = 5,
    blur_mix: float = 1.0,
    noise_std: float = 0.04,
    voxel_dropout: float = 0.08,
) -> np.ndarray:
    losses = torch.empty((data.shape[0],), device=data.device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            target = data[start : start + batch_size]
            model_input = (
                corrupt_fixed(
                    target,
                    seed + start,
                    blur_kernel=blur_kernel,
                    blur_mix=blur_mix,
                    noise_std=noise_std,
                    voxel_dropout=voxel_dropout,
                )
                if corrupted
                else target
            )
            with torch.amp.autocast("cuda", enabled=data.device.type == "cuda"):
                output = model(model_input)
            losses[start : start + target.shape[0]] = per_particle_plain_l2(output["reconstruction"], target)
    return losses.detach().cpu().numpy()


def evaluate(
    model: VoxelPatchMLPAutoencoder,
    data: torch.Tensor,
    *,
    batch_size: int,
    max_batches: int,
    seed: int,
    corrupted: bool,
    blur_kernel: int = 5,
    blur_mix: float = 1.0,
    noise_std: float = 0.04,
    voxel_dropout: float = 0.08,
) -> dict[str, float]:
    rng = torch.Generator(device=data.device)
    rng.manual_seed(seed)
    count = min(data.shape[0], batch_size * max_batches)
    indices = torch.randint(0, data.shape[0], (count,), generator=rng, device=data.device)
    losses = []
    energy_rel = []
    overlap = []
    model.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            target = data[indices[start : start + batch_size]]
            model_input = (
                corrupt_fixed(
                    target,
                    seed + start,
                    blur_kernel=blur_kernel,
                    blur_mix=blur_mix,
                    noise_std=noise_std,
                    voxel_dropout=voxel_dropout,
                )
                if corrupted
                else target
            )
            with torch.amp.autocast("cuda", enabled=data.device.type == "cuda"):
                output = model(model_input)
            recon = output["reconstruction"].float()
            target_f = target.float()
            losses.append(per_particle_plain_l2(recon, target_f))
            target_energy = target_f.sum(dim=(1, 2, 3)).clamp_min(1e-8)
            recon_energy = recon.sum(dim=(1, 2, 3))
            energy_rel.append(torch.abs(recon_energy - target_energy) / target_energy)
            overlap.append(torch.sum(torch.minimum(recon, target_f), dim=(1, 2, 3)) / target_energy)
    loss = torch.cat(losses)
    energy = torch.cat(energy_rel)
    mass_overlap = torch.cat(overlap)
    return {
        "loss": float(loss.mean().detach().cpu()),
        "plain_tensor_l2": float(loss.mean().detach().cpu()),
        "plain_tensor_l2_p50": float(torch.quantile(loss, 0.50).detach().cpu()),
        "plain_tensor_l2_p95": float(torch.quantile(loss, 0.95).detach().cpu()),
        "energy_relative_l1": float(energy.mean().detach().cpu()),
        "mass_overlap": float(mass_overlap.mean().detach().cpu()),
    }


def select_top(losses: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(losses.shape[0] * fraction)))
    selected = np.argpartition(losses, -count)[-count:]
    return selected[np.argsort(losses[selected])[::-1]].astype(np.int64)


def save_model_checkpoint(
    path: Path,
    model: VoxelPatchMLPAutoencoder,
    *,
    model_config: VoxelMLPConfig,
    grid: VoxelGridConfig,
    step: int,
    best_step: int,
    best_val_loss: float,
    extra: dict[str, object] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "grid_config": asdict(grid),
        "model_type": "simple_voxel_patch_mlp_autoencoder_gpu_cached_plain_l2",
        "loss_mode": "plain_tensor_l2",
        "latent_contract": "single direct bottleneck; z_shape only; z_aux has zero dimensions",
        "step": step,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
    }
    if extra:
        payload.update(extra)
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def render_gallery(
    model: VoxelPatchMLPAutoencoder,
    data: torch.Tensor,
    rows: list[dict[str, str]],
    out: Path,
    asset_dir: Path,
    *,
    seed: int,
    blur_kernel: int = 5,
    blur_mix: float = 1.0,
    noise_std: float = 0.04,
    voxel_dropout: float = 0.08,
) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    candidates = [idx for idx, row in enumerate(rows) if int(float(row.get("n_hits", 0))) >= 50]
    rng.shuffle(candidates)
    selected = candidates[:10]
    target = data[torch.as_tensor(selected, device=data.device, dtype=torch.long)]
    noisy = corrupt_fixed(
        target,
        seed,
        blur_kernel=blur_kernel,
        blur_mix=blur_mix,
        noise_std=noise_std,
        voxel_dropout=voxel_dropout,
    )
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=data.device.type == "cuda"):
        output = model(noisy)
    target_np = target.float().detach().cpu().numpy()
    noisy_np = noisy.float().detach().cpu().numpy()
    recon_np = output["reconstruction"].float().detach().cpu().numpy()
    lines = [
        "# Simple z8 Voxel Autoencoder Gallery",
        "",
        "Model input uses the same corruption parameters as training. This model has no explicit transform layer and only an 8D direct latent bottleneck.",
        "",
        f"Corruption: `blur_kernel={blur_kernel}`, `blur_mix={blur_mix}`, `noise_std={noise_std}`, `voxel_dropout={voxel_dropout}`.",
        "",
        "| # | source | particle | hits | energy clean/noisy/recon | relative L2 clean | image |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for out_idx, row_idx in enumerate(selected, start=1):
        row = rows[row_idx]
        image = asset_dir / f"simple_z8_example_{out_idx:02d}.png"
        _render_rows(
            [
                ("clean target", target_np[out_idx - 1]),
                ("noisy input", noisy_np[out_idx - 1]),
                ("reconstruction", recon_np[out_idx - 1]),
                ("abs error", np.abs(recon_np[out_idx - 1] - target_np[out_idx - 1])),
            ],
            image,
            f"{Path(row['source_path']).name} particle {row['particle_id']} hits {row['n_hits']}",
        )
        rel = image.relative_to(out.parent).as_posix()
        l2 = float(
            np.sqrt(np.sum((recon_np[out_idx - 1] - target_np[out_idx - 1]) ** 2))
            / max(np.sqrt(np.sum(target_np[out_idx - 1] ** 2)), 1e-8)
        )
        lines.append(
            f"| {out_idx} | `{row['source_path']}` | {row['particle_id']} | {row['n_hits']} | "
            f"{target_np[out_idx - 1].sum():.3f}/{noisy_np[out_idx - 1].sum():.3f}/{recon_np[out_idx - 1].sum():.3f} | {l2:.4f} | [png]({rel}) |"
        )
    for out_idx in range(1, len(selected) + 1):
        image = asset_dir / f"simple_z8_example_{out_idx:02d}.png"
        lines.extend(["", f"## Example {out_idx}", "", f"![example {out_idx}]({image.relative_to(out.parent).as_posix()})", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple no-transform compressed voxel AE from GPU-resident fp16 tensors.")
    parser.add_argument("--cache", type=Path, default=Path("local_data/processed/phase2_voxel_energy_8x32x32_representative_v001"))
    parser.add_argument("--out", type=Path, default=Path("local_data/experiments/phase2_simple_voxel_ae_z8_b512_v001"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phases", type=int, default=10)
    parser.add_argument("--hard-fraction", type=float, default=0.10)
    parser.add_argument(
        "--mine-once",
        action="store_true",
        help="Select the hard-mined training subset in phase 1 and reuse it in later phases; LR still resets each phase.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--scan-batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--blur-mix", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--voxel-dropout", type=float, default=0.08)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--lr-patience-evals", type=int, default=4)
    parser.add_argument("--stop-lr-below", type=float, default=5e-6)
    parser.add_argument("--eval-interval", type=int, default=512)
    parser.add_argument("--min-steps-per-phase", type=int, default=4096)
    parser.add_argument("--max-steps-per-phase", type=int, default=20512)
    parser.add_argument("--wall-time-limit-s", type=float, default=12 * 60 * 60)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--patch-hidden-dim", type=int, default=192)
    parser.add_argument("--patch-embed-dim", type=int, default=32)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--gallery-out", type=Path, default=Path("experimental_notes/autoencoders/voxel/simple_z8/phase2-simple-voxel-z8-b512-gallery.md"))
    parser.add_argument("--gallery-asset-dir", type=Path, default=Path("experimental_notes/assets/phase2_simple_voxel_z8_b512/gallery_v001"))
    parser.add_argument("--skip-gallery", action="store_true")
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    grid = VoxelGridConfig(t_bins=8, y_bins=32, x_bins=32, time_bin=2.5, xy_bin=2.0)
    train, _ = load_split_to_gpu(args.cache, "train", grid, device=device, dtype=torch.float16, batch_size=4096, seed=args.seed)
    val, _ = load_split_to_gpu(args.cache, "val", grid, device=device, dtype=torch.float16, batch_size=4096, seed=args.seed)
    test, test_rows = load_split_to_gpu(args.cache, "test", grid, device=device, dtype=torch.float16, batch_size=4096, seed=args.seed)
    model_config = VoxelMLPConfig(
        t_bins=8,
        y_bins=32,
        x_bins=32,
        shape_latent_dim=args.latent_dim,
        aux_latent_dim=0,
        hidden_dim=args.hidden_dim,
        patch_hidden_dim=args.patch_hidden_dim,
        patch_embed_dim=args.patch_embed_dim,
        architecture="patch_mlp",
    )
    model = VoxelPatchMLPAutoencoder(model_config).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        checkpoint_config = dict(checkpoint["model_config"])
        expected_config = asdict(model_config)
        mismatches = {
            key: (checkpoint_config.get(key), value)
            for key, value in expected_config.items()
            if checkpoint_config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Init checkpoint model_config does not match requested config: {mismatches}")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded init checkpoint {args.init_checkpoint}", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    metric_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    mining_rows: list[dict[str, object]] = []
    best_metric = float("inf")
    best_state = None
    best_step = 0
    global_step = 0
    fixed_selected_np: np.ndarray | None = None
    fixed_losses: np.ndarray | None = None
    start_time = time.perf_counter()
    for phase in range(1, args.phases + 1):
        if time.perf_counter() - start_time > args.wall_time_limit_s:
            break
        should_scan = not args.mine_once or fixed_selected_np is None
        if should_scan:
            print(f"phase {phase}: scanning {train.shape[0]:,} train particles", flush=True)
            scan_start = time.perf_counter()
            losses = scan_losses(
                model,
                train,
                batch_size=args.scan_batch_size,
                seed=args.seed + phase * 10000,
                corrupted=True,
                blur_kernel=args.blur_kernel,
                blur_mix=args.blur_mix,
                noise_std=args.noise_std,
                voxel_dropout=args.voxel_dropout,
            )
            selected_np = select_top(losses, args.hard_fraction)
            if args.mine_once:
                fixed_selected_np = selected_np
                fixed_losses = losses
            scan_runtime_s = time.perf_counter() - scan_start
        else:
            print(f"phase {phase}: reusing fixed hard-mined subset from phase 1", flush=True)
            scan_start = time.perf_counter()
            losses = fixed_losses
            selected_np = fixed_selected_np
            scan_runtime_s = 0.0
        assert losses is not None
        assert selected_np is not None
        selected = torch.as_tensor(selected_np, device=train.device, dtype=torch.long)
        mining_rows.append(
            {
                "phase": phase,
                "mine_once": bool(args.mine_once),
                "reused_selection": bool(not should_scan),
                "rows_scanned": int(train.shape[0]),
                "selected_rows": int(selected_np.shape[0]),
                "scan_runtime_s": scan_runtime_s,
                "corrupted_l2_mean": float(np.mean(losses)),
                "corrupted_l2_p90": float(np.quantile(losses, 0.90)),
                "selected_corrupted_l2_mean": float(np.mean(losses[selected_np])),
            }
        )
        write_dict_rows(args.out / "mining_summary.csv", mining_rows)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
        current_lr = args.learning_rate
        phase_start = time.perf_counter()
        phase_start_step = global_step + 1
        phase_best = float("inf")
        phase_best_step = global_step
        lr_bad = 0
        evals = 0
        stop_reason = "max_steps"
        rng = torch.Generator(device=train.device)
        rng.manual_seed(args.seed + phase)
        for phase_step in range(1, args.max_steps_per_phase + 1):
            global_step += 1
            batch_indices = selected[torch.randint(0, selected.shape[0], (args.batch_size,), generator=rng, device=train.device)]
            target = train[batch_indices]
            model_input = corrupt_fixed(
                target,
                args.seed + global_step,
                blur_kernel=args.blur_kernel,
                blur_mix=args.blur_mix,
                noise_std=args.noise_std,
                voxel_dropout=args.voxel_dropout,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                output = model(model_input)
                loss, metrics = plain_tensor_l2_loss(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if global_step == 1 or phase_step % args.eval_interval == 0:
                val_metrics = evaluate(
                    model,
                    val,
                    batch_size=args.eval_batch_size,
                    max_batches=args.eval_batches,
                    seed=args.seed + phase * 77,
                    corrupted=True,
                    blur_kernel=args.blur_kernel,
                    blur_mix=args.blur_mix,
                    noise_std=args.noise_std,
                    voxel_dropout=args.voxel_dropout,
                )
                clean_metrics = evaluate(
                    model,
                    val,
                    batch_size=args.eval_batch_size,
                    max_batches=max(1, args.eval_batches // 2),
                    seed=args.seed + phase * 79,
                    corrupted=False,
                )
                monitor = val_metrics["loss"]
                evals += 1
                row = {
                    "step": global_step,
                    "phase": phase,
                    "phase_step": phase_step,
                    "lr": current_lr,
                    **metrics,
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                    **{f"clean_val_{key}": value for key, value in clean_metrics.items()},
                }
                metric_rows.append(row)
                write_dict_rows(args.out / "metrics.csv", metric_rows)
                save_model_checkpoint(
                    args.out / "checkpoint_latest.pt",
                    model,
                    model_config=model_config,
                    grid=grid,
                    step=global_step,
                    best_step=best_step,
                    best_val_loss=best_metric,
                    extra={"phase": phase, "phase_step": phase_step, "val_metrics": val_metrics, "clean_val_metrics": clean_metrics},
                )
                print(
                    f"step {global_step} [simple z{args.latent_dim} phase {phase}]: "
                    f"loss={metrics['loss']:.6g}, val={monitor:.6g}, clean_val={clean_metrics['loss']:.6g}, lr={current_lr:.2e}",
                    flush=True,
                )
                if monitor < phase_best - 1e-5:
                    phase_best = monitor
                    phase_best_step = global_step
                    lr_bad = 0
                    if monitor < best_metric:
                        best_metric = monitor
                        best_step = global_step
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                        save_model_checkpoint(
                            args.out / "checkpoint_best.pt",
                            model,
                            model_config=model_config,
                            grid=grid,
                            step=global_step,
                            best_step=best_step,
                            best_val_loss=best_metric,
                            extra={"phase": phase, "phase_step": phase_step, "val_metrics": val_metrics, "clean_val_metrics": clean_metrics},
                        )
                else:
                    lr_bad += 1
                    if lr_bad >= args.lr_patience_evals:
                        current_lr *= args.lr_decay
                        for group in optimizer.param_groups:
                            group["lr"] = current_lr
                        lr_bad = 0
                        if current_lr < args.stop_lr_below and phase_step >= args.min_steps_per_phase:
                            stop_reason = f"lr_below_{args.stop_lr_below:g}"
                            break
            if time.perf_counter() - start_time > args.wall_time_limit_s:
                stop_reason = "wall_time_limit"
                break
        phase_rows.append(
            {
                "phase": phase,
                "start_step": phase_start_step,
                "end_step": global_step,
                "actual_steps": global_step - phase_start_step + 1,
                "evals": evals,
                "best_step": phase_best_step,
                "best_val_loss": phase_best,
                "stop_reason": stop_reason,
                "final_lr": current_lr,
                "duration_s": time.perf_counter() - phase_start,
            }
        )
        write_dict_rows(args.out / "phase_summary.csv", phase_rows)
        if stop_reason == "wall_time_limit":
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_corrupted = evaluate(
        model,
        test,
        batch_size=args.eval_batch_size,
        max_batches=args.eval_batches,
        seed=args.seed + 9000,
        corrupted=True,
        blur_kernel=args.blur_kernel,
        blur_mix=args.blur_mix,
        noise_std=args.noise_std,
        voxel_dropout=args.voxel_dropout,
    )
    test_clean = evaluate(model, test, batch_size=args.eval_batch_size, max_batches=args.eval_batches, seed=args.seed + 9100, corrupted=False)
    checkpoint = args.out / "checkpoint.pt"
    save_model_checkpoint(
        checkpoint,
        model,
        model_config=model_config,
        grid=grid,
        step=global_step,
        best_step=best_step,
        best_val_loss=best_metric,
        extra={"test_metrics": {"corrupted": test_corrupted, "clean": test_clean}},
    )
    summary = {
        "checkpoint": checkpoint.as_posix(),
        "init_checkpoint": args.init_checkpoint.as_posix() if args.init_checkpoint is not None else None,
        "duration_s": time.perf_counter() - start_time,
        "loss_mode": "plain_tensor_l2",
        "corruption": {
            "blur_kernel": args.blur_kernel,
            "blur_mix": args.blur_mix,
            "noise_std": args.noise_std,
            "voxel_dropout": args.voxel_dropout,
        },
        "best_step": best_step,
        "best_val_loss": best_metric,
        "model_config": asdict(model_config),
        "grid_config": asdict(grid),
        "train_particles": int(train.shape[0]),
        "val_particles": int(val.shape[0]),
        "test_particles": int(test.shape[0]),
        "test_corrupted": test_corrupted,
        "test_clean": test_clean,
        "phase_summary": phase_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not args.skip_gallery:
        render_gallery(
            model,
            test,
            test_rows,
            args.gallery_out,
            args.gallery_asset_dir,
            seed=args.seed,
            blur_kernel=args.blur_kernel,
            blur_mix=args.blur_mix,
            noise_std=args.noise_std,
            voxel_dropout=args.voxel_dropout,
        )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

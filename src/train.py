"""Single-run experiment trainer.
Complies with the specification:
• Full Hydra integration – invoke as module via ``python -m src.train``.
• Handles trial/full mode switching.
• Performs integrity assertions (batch-start, gradient, post-init).
• Comprehensive WandB logging.
• Delegates model definition to ``src.model`` – Baseline/CRUM/ACDM.
• Uses utilities from ``src.preprocess`` for data and few-shot episodes.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim

import wandb

# -----------------------------------------------------------------------------
# Local imports (resolved after Hydra initialises PYTHONPATH) ------------------
# -----------------------------------------------------------------------------
from src.model import build_model, compute_accuracy  # noqa: E402
from src.preprocess import (  # noqa: E402
    build_episode_dataset,
    build_train_val_loaders,
)

# -----------------------------------------------------------------------------
# Reproducibility --------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Assertions -------------------------------------------------------------------


def _assert_gradients(model: nn.Module) -> None:
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), (
        "[assert] No gradients present before optimizer.step(); check loss graph"
    )
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "[assert] All gradients are zero before optimizer.step(); backward cancelled?"
    )


# -----------------------------------------------------------------------------
# Few-shot helpers -------------------------------------------------------------


def _apply_channel_drop(h: torch.Tensor, mode: str, drop_rate: float = 0.3) -> torch.Tensor:
    """Apply random/targeted channel drop during *evaluation* only."""
    if mode == "clean":
        return h
    b, c = h.shape
    if mode == "random":
        mask = (torch.rand(b, c, device=h.device) > drop_rate).float()
    elif mode == "targeted":
        k = max(1, int(drop_rate * c))
        topk = h.abs().topk(k, dim=1, largest=True).indices
        mask = torch.ones_like(h)
        mask.scatter_(1, topk, 0.0)
    else:
        raise ValueError(f"Unknown channel-drop mode: {mode}")
    return h * mask


def _prototypes(support_h: torch.Tensor, support_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    classes = torch.unique(support_y)
    protos = torch.stack([support_h[support_y == c].mean(0) for c in classes], dim=0)
    return protos, classes


def _episode_accuracy(
    model: nn.Module,
    episode: Dict[str, torch.Tensor],
    device: torch.device,
):
    """Compute accuracy for one episode under all corruption modes."""
    shots = int(episode["shots"])
    metrics: Dict[str, float] = {}
    with torch.no_grad():
        model.eval()
        support_h = model.extract_features(episode["support_x"].to(device))
        query_h = model.extract_features(episode["query_x"].to(device))
        support_y = episode["support_y"].to(device)
        query_y = episode["query_y"].to(device)
        for mode in ("clean", "random", "targeted"):
            h_s = _apply_channel_drop(support_h, mode)
            h_q = _apply_channel_drop(query_h, mode)
            protos, proto_labels = _prototypes(h_s, support_y)
            dists = ((h_q.unsqueeze(1) - protos.unsqueeze(0)) ** 2).sum(dim=2)
            preds = proto_labels[dists.argmin(1)]
            acc = (preds == query_y).float().mean().item()
            metrics[f"acc_{mode}_{shots}shot"] = acc
    return metrics


def _fewshot_eval(model: nn.Module, cfg_run: DictConfig, mode: str, device: torch.device):
    ep_dataset = build_episode_dataset(cfg_run, mode)
    if mode == "trial":
        ep_dataset = torch.utils.data.Subset(ep_dataset, list(range(min(15, len(ep_dataset)))))
    loader = torch.utils.data.DataLoader(ep_dataset, batch_size=1, shuffle=False, num_workers=4)

    aggregated: Dict[str, List[float]] = {}
    for batch in loader:
        episode = {k: v.squeeze(0) for k, v in batch.items()}
        res = _episode_accuracy(model, episode, device)
        for k, v in res.items():
            aggregated.setdefault(k, []).append(v)

    metrics = {k: float(torch.tensor(v).mean().item()) for k, v in aggregated.items()}
    return metrics


# -----------------------------------------------------------------------------
# Training helpers -------------------------------------------------------------


def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    cfg_run: DictConfig,
    mode: str,
    wb_run: Optional[wandb.sdk.wandb_run.Run],
) -> None:
    model.train()
    for step, (x, y) in enumerate(loader):
        if mode == "trial" and step >= 2:
            break  # quick validation
        if epoch == 0 and step == 0:
            assert x.size(0) == y.size(0), "[assert] Batch input/label mismatch"
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()
        _assert_gradients(model)
        optimizer.step()

        # logging -----------------------------------------------------------
        if wb_run and step % 10 == 0:
            with torch.no_grad():
                logits = model(x)
                acc = compute_accuracy(logits, y)
            global_step = epoch * len(loader) + step
            wb_run.log(
                {
                    "train/loss": loss.item(),
                    "train/acc": acc,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step,
                },
                commit=True,
            )
    scheduler.step()


def _validate(model: nn.Module, loader, device: torch.device):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = model.loss(x, y)
            logits = model(x)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, total_correct / n


# -----------------------------------------------------------------------------
# Hydra entry-point -------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):  # pragma: no cover
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    # Handle case where cfg.run is a string (from command-line override)
    if isinstance(cfg.run, str):
        run_config_path = Path(get_original_cwd()) / "config" / "runs" / f"{cfg.run}.yaml"
        cfg_run: DictConfig = OmegaConf.load(run_config_path)
    else:
        cfg_run: DictConfig = cfg.run  # alias

    # Automatically adapt settings for *trial* mode -------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg_run.optuna.n_trials = 0
        cfg_run.training.epochs = 1
    else:
        cfg.wandb.mode = "online"

    _seed_everything(int(cfg_run.training.seed))

    # ------------------------- Data ---------------------------------------
    train_loader, val_loader, n_classes = build_train_val_loaders(cfg_run)

    # ------------------------- Model --------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg_run, n_classes).to(device)

    # Post-init integrity check --------------------------------------------
    with torch.no_grad():
        dummy_out = model(torch.randn(2, 3, 32, 32, device=device))
        assert dummy_out.shape[-1] == n_classes, "[assert] Logit dim mismatch"

    # ------------------------- Optimiser & LR sched -----------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(cfg_run.training.learning_rate),
        momentum=0.9,
        weight_decay=float(cfg_run.training.weight_decay),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg_run.training.epochs)
    )

    # ------------------------- WandB --------------------------------------
    wb_run: Optional[wandb.sdk.wandb_run.Run] = None
    if cfg.wandb.mode != "disabled":
        wb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg_run.run_id,
            resume="allow",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"[train] WandB URL: {wb_run.get_url()}")

    # ------------------------- Training loop ------------------------------
    best_val_acc = 0.0
    for epoch in range(int(cfg_run.training.epochs)):
        _train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            cfg_run,
            cfg.mode,
            wb_run,
        )
        val_loss, val_acc = _validate(model, val_loader, device)
        best_val_acc = max(best_val_acc, val_acc)
        if wb_run:
            wb_run.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch}, commit=True)

    # ------------------------- Few-shot evaluation ------------------------
    fewshot_metrics = _fewshot_eval(model, cfg_run, cfg.mode, device)

    # ------------------------- WandB summary ------------------------------
    if wb_run:
        for k, v in fewshot_metrics.items():
            wb_run.summary[k] = v
        wb_run.summary["accuracy"] = fewshot_metrics.get("acc_clean_5shot", best_val_acc)
        wb_run.summary["best_val_acc"] = best_val_acc
        wb_run.finish()

    # ------------------------- Final status -------------------------------
    print(
        f"[train] Completed {cfg_run.run_id} – acc_clean_5shot = "
        f"{fewshot_metrics.get('acc_clean_5shot', 'n/a'):.4f}"
    )


if __name__ == "__main__":
    # Ensure Hydra doesn't change working dir when invoked manually
    sys.argv.append("hydra/job_logging=disabled")
    main()

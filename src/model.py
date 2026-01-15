"""Model definitions: Baseline, CRUM, ACDM.
Referenced by ``train.py``.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from omegaconf import DictConfig

# -----------------------------------------------------------------------------
# Small helpers ----------------------------------------------------------------


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy – returned as *float* (0-1)."""
    preds = logits.argmax(1)
    return float((preds == labels).float().mean().item())


# -----------------------------------------------------------------------------
# Feature extractor ------------------------------------------------------------


def _build_backbone(name: str) -> tuple[nn.Module, int]:
    if name == "resnet18":
        net = tvm.resnet18(weights=None)
        feat_dim = net.fc.in_features
        # remove classification head; we keep conv body and global pool
        net.fc = nn.Identity()
    elif name == "resnet50":
        net = tvm.resnet50(weights=None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone {name}")
    return net, feat_dim


# -----------------------------------------------------------------------------
# Base class -------------------------------------------------------------------

class _BaseModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone_name)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._feat_dim = feat_dim

    # ------------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:  # (B,C)
        h = self.backbone(x)  # (B,C)
        assert h.dim() == 2, "Backbone must output pooled 2-D tensor"
        return h

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.extract_features(x)
        return self.classifier(h)


# -----------------------------------------------------------------------------
# Baseline ---------------------------------------------------------------------

class BaselineModel(_BaseModel):
    def __init__(self, cfg_run: DictConfig, num_classes: int):
        super().__init__(cfg_run.model.name, num_classes)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        return F.cross_entropy(logits, y)


# -----------------------------------------------------------------------------
# CRUM (Uniform Channel Masking) ----------------------------------------------

class CRUMModel(_BaseModel):
    def __init__(self, cfg_run: DictConfig, num_classes: int):
        super().__init__(cfg_run.model.name, num_classes)
        p = cfg_run.training.additional_params
        self.lambda_var = float(p.get("lambda_var", 0.05))
        self.lambda_rand = float(p.get("lambda_rand", 0.1))
        self.temperature = float(p.get("temperature", 2))
        pr = p.get("p_range", [0.3, 0.7])
        self.p_lo, self.p_hi = float(pr[0]), float(pr[1])

    # ------------------------------------------------------------------
    @staticmethod
    def _variance_loss(h: torch.Tensor) -> torch.Tensor:
        mmc = h.abs().mean(0)  # (C,)
        return mmc.var(unbiased=False)

    # ------------------------------------------------------------------
    def _kl(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        return (
            F.kl_div(
                F.log_softmax(p_logits / T, dim=1),
                F.softmax(q_logits / T, dim=1),
                reduction="batchmean",
            )
            * (T**2)
        )

    # ------------------------------------------------------------------
    def _random_mask(self, h: torch.Tensor) -> torch.Tensor:
        drop_rate = float(torch.empty(1).uniform_(self.p_lo, self.p_hi))
        mask = (torch.rand_like(h) > drop_rate).float()
        return h * mask

    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.extract_features(x)
        logits_clean = self.classifier(h)
        loss_ce = F.cross_entropy(logits_clean, y)
        loss_var = self._variance_loss(h)
        h_rand = self._random_mask(h)
        logits_rand = self.classifier(h_rand)
        loss_rand = self._kl(logits_clean, logits_rand)
        return loss_ce + self.lambda_var * loss_var + self.lambda_rand * loss_rand


# -----------------------------------------------------------------------------
# ACDM (Proposed) --------------------------------------------------------------

class ACDMModel(CRUMModel):  # re-use helpers
    def __init__(self, cfg_run: DictConfig, num_classes: int):
        super().__init__(cfg_run, num_classes)
        p = cfg_run.training.additional_params
        self.lambda_adv = float(p.get("lambda_adv", 0.3))
        self.enable_mix = True

    # ------------------------------------------------------------------
    def _adversarial_mask(self, h: torch.Tensor, drop_rate: float) -> torch.Tensor:
        B, C = h.shape
        k = max(1, int(drop_rate * C))
        topk = h.abs().topk(k, dim=1, largest=True).indices
        mask = torch.ones_like(h)
        mask.scatter_(1, topk, 0.0)
        return h * mask

    # ------------------------------------------------------------------
    def _channel_mix(self, h: torch.Tensor) -> torch.Tensor:
        B = h.size(0)
        perm = torch.randperm(B, device=h.device)
        alpha = torch.distributions.Beta(2, 2).sample((B, 1)).to(h.device)
        return alpha * h + (1 - alpha) * h[perm]

    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.extract_features(x)
        logits_clean = self.classifier(h)
        loss_ce = F.cross_entropy(logits_clean, y)
        loss_var = self._variance_loss(h)

        # determine drop rate p ~ U(p_lo, p_hi)
        drop_rate = float(torch.empty(1).uniform_(self.p_lo, self.p_hi))

        # random mask term --------------------------------------------------
        h_rand = self._random_mask(h)
        logits_rand = self.classifier(h_rand)
        loss_rand = self._kl(logits_clean, logits_rand)

        # adversarial mask term -------------------------------------------
        h_adv = self._adversarial_mask(h, drop_rate)
        logits_adv = self.classifier(h_adv)
        loss_adv = self._kl(logits_clean, logits_adv)

        # optional channel mix -------------------------------------------
        loss_mix = 0.0
        if self.enable_mix and torch.rand(1).item() < 0.5:
            h_mix = self._channel_mix(h)
            logits_mix = self.classifier(h_mix)
            loss_mix = F.cross_entropy(logits_mix, y)

        return (
            loss_ce
            + self.lambda_var * loss_var
            + self.lambda_rand * loss_rand
            + self.lambda_adv * loss_adv
            + 0.5 * loss_mix  # small weight for mixup regulariser
        )


# -----------------------------------------------------------------------------
# Public factory ---------------------------------------------------------------


def build_model(cfg_run: DictConfig, num_classes: int) -> nn.Module:
    method = cfg_run.method.lower()
    if "acdm" in method:
        return ACDMModel(cfg_run, num_classes)
    if "crum" in method:
        return CRUMModel(cfg_run, num_classes)
    return BaselineModel(cfg_run, num_classes)

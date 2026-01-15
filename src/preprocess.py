"""Data loading & episode generation for CIFAR-FS.
Strictly no label leakage – labels are *never* concatenated to inputs.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR100

# -----------------------------------------------------------------------------
# Normalisation constants ------------------------------------------------------

_CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR_STD = (0.2675, 0.2565, 0.2761)

# -----------------------------------------------------------------------------
# Utilities --------------------------------------------------------------------


def _cifarfs_split() -> Tuple[List[int], List[int]]:
    classes = list(range(100))
    random.Random(123).shuffle(classes)  # fixed split for reproducibility
    return classes[:64], classes[64:]


def _build_transforms(cfg_run: DictConfig, train: bool) -> T.Compose:  # noqa: D401
    ops: List[T.transforms] = []
    if train:
        for entry in cfg_run.dataset.preprocessing:
            key, val = next(iter(entry.items()))
            if key == "random_crop":
                ops.append(T.RandomCrop(val, padding=4))
            elif key == "random_flip" and val == "horizontal":
                ops.append(T.RandomHorizontalFlip())
    ops.extend([T.ToTensor(), T.Normalize(_CIFAR_MEAN, _CIFAR_STD)])
    return T.Compose(ops)


# -----------------------------------------------------------------------------
# Factory: train / val loaders -------------------------------------------------


class _RemappedDataset(Dataset):
    """Wraps a dataset and remaps labels to contiguous range [0, n_classes-1]."""
    def __init__(self, base_dataset, class_to_idx):
        self.base = base_dataset
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        # Remap label to contiguous index
        new_label = self.class_to_idx[label]
        return img, new_label


def build_train_val_loaders(cfg_run: DictConfig):
    root = Path(".cache") / "datasets" / "cifar100"
    root.mkdir(parents=True, exist_ok=True)

    ds_full = CIFAR100(
        root=str(root),
        train=True,
        download=True,
        transform=_build_transforms(cfg_run, train=True),
    )

    train_classes, _ = _cifarfs_split()
    # Create mapping from original class labels to contiguous indices [0, 63]
    class_to_idx = {cls: idx for idx, cls in enumerate(train_classes)}

    idx = [i for i, t in enumerate(ds_full.targets) if t in train_classes]
    ds_sub = Subset(ds_full, idx)

    # Wrap with remapping
    ds_remapped = _RemappedDataset(ds_sub, class_to_idx)

    val_ratio = 0.1
    val_size = int(len(ds_remapped) * val_ratio)
    train_size = len(ds_remapped) - val_size
    gen = torch.Generator().manual_seed(int(cfg_run.training.seed))
    train_set, val_set = random_split(ds_remapped, [train_size, val_size], generator=gen)

    loader_train = DataLoader(
        train_set,
        batch_size=int(cfg_run.training.batch_size),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    loader_val = DataLoader(
        val_set,
        batch_size=int(cfg_run.training.batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader_train, loader_val, len(train_classes)


# -----------------------------------------------------------------------------
# Episode dataset for meta-testing --------------------------------------------

class CIFARFSEpisodeDataset(Dataset):
    """Pre-generated few-shot episodes."""

    def __init__(self, cfg_run: DictConfig, mode: str, n_way: int, shots: int, query: int = 15):
        super().__init__()
        self.cfg = cfg_run
        self.mode = mode
        self.n_way = n_way
        self.shots = shots
        self.query = query

        root = Path(".cache") / "datasets" / "cifar100"
        self.base = CIFAR100(
            root=str(root),
            train=True,
            download=True,
            transform=_build_transforms(cfg_run, train=False),
        )
        _, self.test_classes = _cifarfs_split()

        # map class -> indices
        self.cls2idx: Dict[int, List[int]] = {}
        for i, t in enumerate(self.base.targets):
            if t in self.test_classes:
                self.cls2idx.setdefault(t, []).append(i)

        self.episodes: List[Tuple[List[int], List[int]]] = []
        self._pre_sample_episodes()

    # ------------------------------------------------------------------
    def _pre_sample_episodes(self):
        rng = random.Random(int(self.cfg.training.seed))
        n_episodes = 600 if self.mode == "full" else 10
        for _ in range(n_episodes):
            cls = rng.sample(self.test_classes, self.n_way)
            support_idx, query_idx = [], []
            for c in cls:
                idx_pool = rng.sample(self.cls2idx[c], self.shots + self.query)
                support_idx.extend(idx_pool[: self.shots])
                query_idx.extend(idx_pool[self.shots :])
            self.episodes.append((support_idx, query_idx))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        support_idx, query_idx = self.episodes[idx]
        s_imgs = torch.stack([self.base[i][0] for i in support_idx])
        q_imgs = torch.stack([self.base[i][0] for i in query_idx])
        s_labels = torch.tensor([self.base[i][1] for i in support_idx], dtype=torch.long)
        q_labels = torch.tensor([self.base[i][1] for i in query_idx], dtype=torch.long)

        # map to episode-local labels 0..n_way-1
        unique = torch.unique(s_labels)
        mapper = {int(c): i for i, c in enumerate(unique.tolist())}
        s_labels = torch.tensor([mapper[int(t)] for t in s_labels])
        q_labels = torch.tensor([mapper[int(t)] for t in q_labels])

        return {
            "support_x": s_imgs,
            "support_y": s_labels,
            "query_x": q_imgs,
            "query_y": q_labels,
            "shots": self.shots,
        }


# -----------------------------------------------------------------------------
# Public factory ---------------------------------------------------------------


def build_episode_dataset(cfg_run: DictConfig, mode: str):
    ds = [CIFARFSEpisodeDataset(cfg_run, mode, 5, s) for s in (1, 5, 20)]
    return ConcatDataset(ds)

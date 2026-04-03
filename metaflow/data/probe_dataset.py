from typing import Tuple
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms


def _load_cifar10(root="./data", train=True):
    """Load CIFAR-10 with proper normalization and optional augmentation."""
    if train:
        # Training: add data augmentation
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])
    else:
        # Test: only normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])
    return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)


def get_probe_dataset(root="./data", size: int = 5000, seed: int = 42) -> Dataset:
    """
    Probe distribution Q(x):
    - no private samples are needed; here we just use a random subset of CIFAR-10 train.
    - in a real system this could be synthetic or public data.
    """
    full = _load_cifar10(root=root, train=True)
    if size >= len(full):
        return full
    g = torch.Generator()
    g.manual_seed(seed)
    probe, _ = random_split(full, [size, len(full) - size], generator=g)
    return probe


def get_probe_splits(
    root="./data",
    router_size: int = 5000,
    distill_size: int = 5000,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Return two non-overlapping subsets from CIFAR-10 train:
    - router subset (for coordinator/router fitting)
    - distill subset (for student distillation)
    """
    full = _load_cifar10(root=root, train=True)
    if router_size + distill_size > len(full):
        raise ValueError(
            f"router_size + distill_size exceeds CIFAR-10 train size ({len(full)})."
        )

    g = torch.Generator()
    g.manual_seed(seed)

    router_ds, distill_ds, _ = random_split(
        full,
        [router_size, distill_size, len(full) - router_size - distill_size],
        generator=g,
    )
    return router_ds, distill_ds


def get_test_dataset(root="./data") -> Dataset:
    """
    Held-out test set (standard CIFAR-10 test).
    """
    return _load_cifar10(root=root, train=False)

from typing import Tuple
import torch
from torch.utils.data import Dataset, Subset, random_split

from . import _load_mnist


def get_probe_dataset(root="./data", size: int = 5000, seed: int = 42) -> Dataset:
    """
    Probe distribution Q(x):
    - no private samples are needed; here we just use a random subset of MNIST train.
    - in a real system this could be synthetic or public data.
    """
    full = _load_mnist(root=root, train=True)
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
    Return two non-overlapping subsets from MNIST train:
    - router subset (for coordinator/router fitting)
    - distill subset (for student distillation)
    """
    full = _load_mnist(root=root, train=True)
    if router_size + distill_size > len(full):
        raise ValueError(
            f"router_size + distill_size exceeds MNIST train size ({len(full)})."
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
    Held-out test set (standard MNIST test).
    """
    return _load_mnist(root=root, train=False)

from typing import Tuple
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms


def _load_mnist(root="./data", train=True):
    transform = transforms.ToTensor()
    return datasets.MNIST(root=root, train=train, download=True, transform=transform)


def get_probe_dataset(root="./data", size: int = 5000) -> Dataset:
    """
    Probe distribution Q(x):
    - no private samples are needed; here we just use a random subset of MNIST train.
    - in a real system this could be synthetic or public data.
    """
    full = _load_mnist(root=root, train=True)
    if size >= len(full):
        return full
    probe, _ = random_split(full, [size, len(full) - size])
    return probe


def get_test_dataset(root="./data") -> Dataset:
    """
    Held-out test set (standard MNIST test).
    """
    return _load_mnist(root=root, train=False)

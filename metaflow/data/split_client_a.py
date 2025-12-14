import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


def _load_mnist(root="./data", train=True):
    transform = transforms.ToTensor()
    return datasets.MNIST(root=root, train=train, download=True, transform=transform)


def get_client_a_dataset(root="./data", train=True) -> Dataset:
    """
    Client A: sees digits 0-4.
    """
    full = _load_mnist(root=root, train=train)
    indices = [i for i, (x, y) in enumerate(full) if y in {0, 1, 2, 3, 4}]
    return Subset(full, indices)

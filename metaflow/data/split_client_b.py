# metaflow/data/split_client_b.py
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from .split_client_a import _load_cifar10


def get_client_b_dataset(root="./data", train=True):
    """Client B: all animals including overlap.
    
    CIFAR-10 classes:
    2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse
    Overlaps with Client A on bird(2) for realistic routing challenge.
    """
    full = _load_cifar10(root=root, train=train)
    allowed = {2, 3, 4, 5, 6, 7}  # animals including overlap
    indices = [i for i, (_, y) in enumerate(full) if y in allowed]
    return Subset(full, indices)




if __name__ == "__main__":
    loader = get_client_b_dataset()
    x, y = next(iter(loader))
    print("Client B batch:", x.shape, y[:10])

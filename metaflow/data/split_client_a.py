import torch
from torch.utils.data import Dataset, Subset

from . import _load_mnist


def get_client_a_dataset(root="./data", train=True):
    full = _load_mnist(root=root, train=train)
    indices = [i for i, (_, y) in enumerate(full) if y in {0,1,2,3,4}]
    return Subset(full, indices)



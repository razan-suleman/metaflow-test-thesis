# metaflow/data/split_client_b.py
import torch
from torch.utils.data import Subset

from . import _load_mnist


def get_client_b_dataset(root="./data", train=True):
    full = _load_mnist(root=root, train=train)
    indices = [i for i, (_, y) in enumerate(full) if y in {3,4,5,6,7,8,9}]
    return Subset(full, indices)





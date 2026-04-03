import torch
from torch.utils.data import Dataset, Subset
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


def get_client_a_dataset(root="./data", train=True):
    """Client A: mostly vehicles with some overlap.
    
    CIFAR-10 classes:
    0-airplane, 1-automobile, 2-bird, 8-ship, 9-truck
    Overlaps with Client B on bird(2) for realistic routing challenge.
    """
    full = _load_cifar10(root=root, train=train)
    allowed = {0, 1, 2, 8, 9}  # vehicles + bird for overlap
    indices = [i for i, (_, y) in enumerate(full) if y in allowed]
    return Subset(full, indices)



# metaflow/data/split_client_b.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_client_b_loader(batch_size: int = 64, root: str = "./data") -> DataLoader:
    """
    Client B: non-IID subset of MNIST, digits 5-9.
    """
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)

    # Filter digits 5-9
    indices = [i for i, (_, y) in enumerate(train_dataset) if y in [5, 6, 7, 8, 9]]
    subset = Subset(train_dataset, indices)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader


if __name__ == "__main__":
    loader = get_client_b_loader()
    x, y = next(iter(loader))
    print("Client B batch:", x.shape, y[:10])

from torchvision import datasets, transforms


def _load_mnist(root="./data", train=True):
    """Load MNIST dataset with standard ToTensor transform."""
    transform = transforms.ToTensor()
    return datasets.MNIST(root=root, train=train, download=True, transform=transform)


from .split_client_a import get_client_a_dataset
from .split_client_b import get_client_b_dataset
from .probe_dataset import get_probe_dataset, get_test_dataset, get_probe_splits

import torch
from torch.utils.data import Dataset

class NoisyLabelDataset(Dataset):
    def __init__(self, base: Dataset, num_classes: int = 10, noise_p: float = 0.0, seed: int = 0):
        self.base = base
        self.num_classes = num_classes
        self.noise_p = float(noise_p)
        g = torch.Generator().manual_seed(seed)

        n = len(base)
        flip_mask = torch.rand(n, generator=g) < self.noise_p
        self.flip_mask = flip_mask

        # pre-sample random wrong labels
        self.new_labels = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.noise_p > 0 and self.flip_mask[idx]:
            y = int(self.new_labels[idx].item())
        return x, y

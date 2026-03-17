import os
import random
from typing import Literal
from data.noisy_wrapper import NoisyLabelDataset

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data import get_client_a_dataset, get_client_b_dataset

from models import local_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"


def get_client_dataset(client: Literal["a", "b"]):
    if client == "a":
        return get_client_a_dataset()
    elif client == "b":
        return get_client_b_dataset()
    else:
        raise ValueError("client must be 'a' or 'b'")


def train_client(client: str, epochs: int = 3, batch_size: int = 64, lr: float = 1e-3, seed: int = 42):
    # ensure determenisitic behaivior like weight initialization
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = get_client_dataset(client)
    print(f"Client {client} train size: {len(dataset)}")

    # add noise 
    if client == "b":
        dataset = NoisyLabelDataset(dataset, num_classes=10, noise_p=0.2, seed=0)

    g = torch.Generator()
    g.manual_seed(seed) # every time you use g, it will produce the same random sequence.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    model = local_model.LocalCNN().to(DEVICE) # move the model to the same hardware
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad() # clears the gradients from the previous step.    
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Client {client} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"client_{client}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved client {client} model to {ckpt_path}")


if __name__ == "__main__":
    train_client("a")
    train_client("b")

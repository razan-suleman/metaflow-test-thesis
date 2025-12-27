import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from metaflow.data import get_probe_dataset
from metaflow.models.student_model import StudentModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"


def train_student(epochs: int = 3, batch_size: int = 64, lr: float = 1e-3, probe_size: int = 5000):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = get_probe_dataset(size=probe_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = StudentModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Student | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, "student.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved student model to {ckpt_path}")


if __name__ == "__main__":
    train_student(epochs=1)

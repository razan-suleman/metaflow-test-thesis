import os
import torch
from torch.utils.data import DataLoader

from metaflow.data import get_test_dataset
from metaflow.models.local_model import LocalCNN
from metaflow.models.student_model import StudentModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"


def _evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def load_client_model(client: str) -> LocalCNN:
    model = LocalCNN().to(DEVICE)
    state = torch.load(os.path.join(CHECKPOINT_DIR, f"client_{client}.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    return model


def load_student_model() -> StudentModel:
    model = StudentModel().to(DEVICE)
    state = torch.load(os.path.join(CHECKPOINT_DIR, "student.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    return model


if __name__ == "__main__":
    test_ds = get_test_dataset()
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    client_a = load_client_model("a")
    client_b = load_client_model("b")
    student = load_student_model()

    acc_a = _evaluate_model(client_a, test_loader)
    acc_b = _evaluate_model(client_b, test_loader)
    acc_s = _evaluate_model(student, test_loader)

    print(f"Client A accuracy: {acc_a:.4%}")
    print(f"Client B accuracy: {acc_b:.4%}")
    print(f"Student (MetaFlow) accuracy: {acc_s:.4%}")

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import get_probe_dataset
from models.student_model import StudentModel
from core import MetaFlow
from coordinators.confidence_select import ConfidenceSelectCoordinator
from agents.local_cnn_agent import LocalCNNAgent
from evaluate import load_client_model, DEVICE

CHECKPOINT_DIR = "checkpoints"


def distill(
    epochs=5,
    batch_size=128,
    lr=1e-3,
    temperature=2.0,
    coordinator=None,  # instance implementing Coordinator protocol
    seed=42,
):
    """Distill a student model using the MetaFlow teacher.

    Parameters mirror the old script. If ``coordinator`` is None a
    :class:`ConfidenceSelectCoordinator` is constructed.  Pass a different
    coordinator instance to vary the teacher behaviour.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -------- Teacher: MetaFlow --------
    a = load_client_model("a")
    b = load_client_model("b")

    if coordinator is None:
        coordinator = ConfidenceSelectCoordinator()

    teacher = MetaFlow(
        agents=[
            LocalCNNAgent(a, DEVICE),
            LocalCNNAgent(b, DEVICE),
        ],
        coordinator=coordinator,
    )

    # -------- Student --------
    student = StudentModel().to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    # use probe dataset for distillation so student sees the same inputs used in other parts
    dataset = get_probe_dataset(seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    student.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher.predict_logits(x)

            student_logits = student(x)

            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Distill loss: {avg_loss:.4f}")

    torch.save(
        student.state_dict(),
        os.path.join(CHECKPOINT_DIR, "student.pt"),
    )
    print("Saved distilled student to checkpoints/student.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run distillation to create a student model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--coordinator",
        choices=["confidence", "margin"],
        default="confidence",
        help="Which coordinator to use for the teacher MetaFlow.",
    )
    args = parser.parse_args()

    if args.coordinator == "confidence":
        coord = ConfidenceSelectCoordinator()
    else:
        coord = MarginSelectCoordinator()
    distill(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        coordinator=coord,
    )

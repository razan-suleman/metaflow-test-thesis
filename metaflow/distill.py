import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metaflow.data import get_test_dataset  # replace with probe dataset later
from metaflow.models.student_model import StudentModel
from metaflow.core import MetaFlow
from metaflow.coordinators.confidence_select import ConfidenceSelectCoordinator
from metaflow.agents.local_cnn_agent import LocalCNNAgent
from metaflow.evaluate import load_client_model, DEVICE

CHECKPOINT_DIR = "checkpoints"


def distill(
    epochs=5,
    batch_size=128,
    lr=1e-3,
    temperature=2.0,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # -------- Teacher: MetaFlow --------
    a = load_client_model("a")
    b = load_client_model("b")

    teacher = MetaFlow(
        agents=[
            LocalCNNAgent(a, DEVICE),
            LocalCNNAgent(b, DEVICE),
        ],
        coordinator=ConfidenceSelectCoordinator(),
    )

    # -------- Student --------
    student = StudentModel().to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    dataset = get_test_dataset()  # swap with probe dataset later
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    distill()

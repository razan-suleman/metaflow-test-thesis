import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from metaflow.train_local import train_client
from metaflow.distill import distill
from metaflow.evaluate import (
    load_client_model,
    load_student_model,
    _evaluate_model,
    DEVICE,
)
from metaflow.core import MetaFlow
from metaflow.agents.local_cnn_agent import LocalCNNAgent
from metaflow.coordinators.confidence_select import ConfidenceSelectCoordinator
from metaflow.coordinators.margin_select import MarginSelectCoordinator
from metaflow.data import get_test_dataset

# directories used by the pipeline
ARTIFACTS_DIR = "artifacts"
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")
CHECKPOINT_DIR = "checkpoints"


def ensure_dirs():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ARTIFACTS_DIR, "probe_logits"), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_metaflow(system: MetaFlow, loader: DataLoader) -> float:
    """Return accuracy of a MetaFlow system on given loader."""
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            y = y.to(DEVICE)
            logits = system.predict_logits(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Run full MetaFlow pipeline: train clients, distill student, evaluate, and save metrics."
    )
    parser.add_argument("--exp-name", required=True, help="Unique name for this experiment; results written to artifacts/results/<exp_name>.json")
    parser.add_argument(
        "--coordinator",
        choices=["confidence", "margin"],
        default="confidence",
        help="Which coordinator to use when building the teacher MetaFlow.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="If provided, the script will not retrain client models; existing checkpoints must be present.",
    )
    parser.add_argument(
        "--distill-epochs",
        type=int,
        default=5,
        help="Number of epochs to run distillation for the student.",
    )
    args = parser.parse_args()

    ensure_dirs()

    # train or verify clients
    for client in ["a", "b"]:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"client_{client}.pt")
        if args.skip-train:
            if not os.path.exists(ckpt_path):
                raise RuntimeError(
                    f"Requested --skip-train but checkpoint for client {client} is missing ({ckpt_path})."
                )
            print(f"Skipping training for client {client}, checkpoint exists.")
        else:
            # retrain unconditionally; train_client itself recreates the checkpoint folder.
            train_client(client)

    # build teacher MetaFlow
    a_model = load_client_model("a")
    b_model = load_client_model("b")
    if args.coordinator == "confidence":
        coord = ConfidenceSelectCoordinator()
    else:
        coord = MarginSelectCoordinator()

    teacher = MetaFlow(
        agents=[LocalCNNAgent(a_model, DEVICE), LocalCNNAgent(b_model, DEVICE)],
        coordinator=coord,
    )

    # distill student
    distill(epochs=args.distill_epochs, coordinator=coord)

    # evaluation
    test_ds = get_test_dataset()
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    acc_a = _evaluate_model(a_model, test_loader)
    acc_b = _evaluate_model(b_model, test_loader)
    acc_teacher = evaluate_metaflow(teacher, test_loader)
    student_model = load_student_model()
    acc_student = _evaluate_model(student_model, test_loader)

    metrics = {
        "exp_name": args.exp_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "coordinator": args.coordinator,
        "acc_client_a": acc_a,
        "acc_client_b": acc_b,
        "acc_teacher": acc_teacher,
        "acc_student": acc_student,
        "coordinator_stats": getattr(coord, "stats", None),
    }

    out_path = os.path.join(RESULTS_DIR, f"{args.exp_name}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Pipeline complete; saved results to {out_path}")


if __name__ == "__main__":
    main()

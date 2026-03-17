import argparse
import json
import os
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from train_local import train_client
from distill import distill
from evaluate import (
    load_client_model,
    load_student_model,
    _evaluate_model,
    DEVICE,
)
from core import MetaFlow
from agents.local_cnn_agent import LocalCNNAgent
from coordinators.confidence_select import ConfidenceSelectCoordinator
from coordinators.margin_select import MarginSelectCoordinator
from coordinators.average_logits import AverageLogitsCoordinator
from coordinators.agree_then_router import AgreeThenRouterCoordinator
from data import get_test_dataset, get_probe_dataset, get_probe_splits

# directories used by the pipeline
ARTIFACTS_DIR = "artifacts"
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")
CHECKPOINT_DIR = "checkpoints"

# default experiment settings (used when no CLI flags are provided)
DEFAULT_EXP_NAME = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_COORDINATOR = "confidence"
DEFAULT_SKIP_TRAIN = False
DEFAULT_DISTILL_EPOCHS = 7
DEFAULT_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def reset_selection_stats(coordinator) -> None:
    if not hasattr(coordinator, "stats"):
        return
    stats = coordinator.stats
    if not isinstance(stats, dict):
        return
    if "picked" in stats and isinstance(stats["picked"], list):
        stats["picked"] = [0] * len(stats["picked"])
    if "total" in stats:
        stats["total"] = 0
    if "disagree" in stats:
        stats["disagree"] = 0


def main():
    parser = argparse.ArgumentParser(
        description="Run full MetaFlow pipeline: train clients, distill student, evaluate, and save metrics."
    )
    parser.add_argument(
        "--exp-name",
        default=DEFAULT_EXP_NAME,
        help="Unique name for this experiment; results written to artifacts/results/<exp_name>.json",
    )
    parser.add_argument(
        "--coordinator",
        choices=["confidence", "margin", "average", "agree_router"],
        default=DEFAULT_COORDINATOR,
        help="Which coordinator to use when building the teacher MetaFlow.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=DEFAULT_SKIP_TRAIN,
        help="If provided, the script will not retrain client models; existing checkpoints must be present.",
    )
    parser.add_argument(
        "--distill-epochs",
        type=int,
        default=DEFAULT_DISTILL_EPOCHS,
        help="Number of epochs to run distillation for the student.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Global random seed for reproducibility.",
    )
    args = parser.parse_args()

    print(
        f"Running with settings: exp_name={args.exp_name}, "
        f"coordinator={args.coordinator}, skip_train={args.skip_train}, "
        f"distill_epochs={args.distill_epochs}, seed={args.seed}"
    )

    set_seed(args.seed)
    ensure_dirs()

    # train or verify clients
    for client in ["a", "b"]:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"client_{client}.pt")
        if args.skip_train:
            if not os.path.exists(ckpt_path):
                raise RuntimeError(
                    f"Requested --skip-train but checkpoint for client {client} is missing ({ckpt_path})."
                )
            print(f"Skipping training for client {client}, checkpoint exists.")
        else:
            # retrain unconditionally; train_client itself recreates the checkpoint folder.
            train_client(client, seed=args.seed)

    # build teacher MetaFlow
    a_model = load_client_model("a")
    b_model = load_client_model("b")
    distill_probe_ds = None
    if args.coordinator == "confidence":
        coord = ConfidenceSelectCoordinator()
    elif args.coordinator == "margin":
        coord = MarginSelectCoordinator()
    elif args.coordinator == "agree_router":
        coord = AgreeThenRouterCoordinator()
        router_probe_ds, distill_probe_ds = get_probe_splits(seed=args.seed)
        coord.fit_from_models(
            a_model,
            b_model,
            router_probe_ds,
            DEVICE,
            seed=args.seed,
        )
        print(
            "Trained agree_router on probe disagreements: "
            f"samples={coord.stats['router_train_samples']}, "
            f"pos_rate={coord.stats['router_train_pos_rate']}, "
            f"val_acc={coord.stats.get('router_val_acc')}, "
            f"margin_val_acc={coord.stats.get('router_margin_baseline_acc')}, "
            f"enabled={coord.stats.get('router_enabled')}"
        )
    else:
        coord = AverageLogitsCoordinator()
    if distill_probe_ds is None:
        distill_probe_ds = get_probe_dataset(seed=args.seed)

    teacher = MetaFlow(
        agents=[LocalCNNAgent(a_model, DEVICE), LocalCNNAgent(b_model, DEVICE)],
        coordinator=coord,
    )

    # distill student
    distill(
        epochs=args.distill_epochs,
        coordinator=coord,
        seed=args.seed,
        probe_dataset=distill_probe_ds,
    )

    # evaluation
    reset_selection_stats(coord)
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
        "seed": args.seed,
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

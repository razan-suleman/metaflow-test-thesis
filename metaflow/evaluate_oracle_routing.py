"""
Evaluate oracle-routing upper bound for two experts.

Definition:
    oracle_correct = (pred_A == y) OR (pred_B == y)
    oracle_acc = mean(oracle_correct)

This script also reports practical headroom versus current coordinators.
"""

import torch
from torch.utils.data import DataLoader

from data import get_test_dataset
from evaluate import load_client_model, _evaluate_model, DEVICE
from core import MetaFlow
from agents.local_cnn_agent import LocalCNNAgent
from coordinators.confidence_select import ConfidenceSelectCoordinator
from coordinators.margin_select import MarginSelectCoordinator
from coordinators.average_logits import AverageLogitsCoordinator


def evaluate_system(system: MetaFlow, loader: DataLoader) -> float:
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            y = y.to(DEVICE)
            logits = system.predict_logits(x)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_oracle(a_model, b_model, loader: DataLoader) -> float:
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred_a = a_model(x).argmax(dim=-1)
            pred_b = b_model(x).argmax(dim=-1)

            oracle_correct = (pred_a == y) | (pred_b == y)
            correct += oracle_correct.sum().item()
            total += y.size(0)
    return correct / total


def main():
    test_loader = DataLoader(get_test_dataset(), batch_size=128, shuffle=False)

    a_model = load_client_model("a")
    b_model = load_client_model("b")

    acc_a = _evaluate_model(a_model, test_loader)
    acc_b = _evaluate_model(b_model, test_loader)
    acc_oracle = evaluate_oracle(a_model, b_model, test_loader)

    confidence = MetaFlow(
        agents=[LocalCNNAgent(a_model, DEVICE), LocalCNNAgent(b_model, DEVICE)],
        coordinator=ConfidenceSelectCoordinator(),
    )
    margin = MetaFlow(
        agents=[LocalCNNAgent(a_model, DEVICE), LocalCNNAgent(b_model, DEVICE)],
        coordinator=MarginSelectCoordinator(),
    )
    average = MetaFlow(
        agents=[LocalCNNAgent(a_model, DEVICE), LocalCNNAgent(b_model, DEVICE)],
        coordinator=AverageLogitsCoordinator(),
    )

    acc_conf = evaluate_system(confidence, test_loader)
    acc_margin = evaluate_system(margin, test_loader)
    acc_avg = evaluate_system(average, test_loader)

    print("\n=== Oracle Routing Upper Bound (CIFAR-10 test) ===\n")
    print(f"Client A accuracy        : {acc_a:.4%}")
    print(f"Client B accuracy        : {acc_b:.4%}")
    print(f"Oracle routing accuracy  : {acc_oracle:.4%}")
    print()
    print("Current coordinators:")
    print(f"  Confidence             : {acc_conf:.4%}")
    print(f"  Margin                 : {acc_margin:.4%}")
    print(f"  Average logits         : {acc_avg:.4%}")
    print()
    print("Headroom to oracle:")
    print(f"  vs Margin              : {(acc_oracle - acc_margin):+.4%}")
    print(f"  vs Confidence          : {(acc_oracle - acc_conf):+.4%}")
    print(f"  vs Best client         : {(acc_oracle - max(acc_a, acc_b)):+.4%}")


if __name__ == "__main__":
    main()

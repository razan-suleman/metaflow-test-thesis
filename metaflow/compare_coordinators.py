"""
compare_coordinators.py
Evaluate MetaFlow teacher accuracy for all three coordinator strategies
using existing client checkpoints. No training or distillation needed.
"""
import torch
from torch.utils.data import DataLoader

from evaluate import load_client_model, _evaluate_model, DEVICE
from core import MetaFlow
from agents.local_cnn_agent import LocalCNNAgent
from coordinators.confidence_select import ConfidenceSelectCoordinator
from coordinators.margin_select import MarginSelectCoordinator
from coordinators.average_logits import AverageLogitsCoordinator
from coordinators.agree_then_router import AgreeThenRouterCoordinator
from coordinators.neural_router import create_neural_router_coordinator
from data import get_test_dataset, get_probe_dataset


def evaluate_system(system, loader):
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
    test_loader = DataLoader(get_test_dataset(), batch_size=128, shuffle=False)

    a_model = load_client_model("a")
    b_model = load_client_model("b")

    # individual client baselines
    acc_a = _evaluate_model(a_model, test_loader)
    acc_b = _evaluate_model(b_model, test_loader)

    coordinators = {
        "Confidence": ConfidenceSelectCoordinator(),
        "Margin":     MarginSelectCoordinator(),
        "Average":    AverageLogitsCoordinator(),
    }

    agree_router = AgreeThenRouterCoordinator()
    agree_router.fit_from_models(
        a_model,
        b_model,
        get_probe_dataset(seed=42),
        DEVICE,
        seed=42,
    )
    coordinators["AgreeRouter"] = agree_router
    
    # Train neural router coordinator
    probe_data = get_probe_dataset(seed=42)
    probe_loader = DataLoader(probe_data, batch_size=64, shuffle=False)
    neural_router_coord = create_neural_router_coordinator(
        a_model, b_model, probe_loader, device=DEVICE, epochs=50
    )
    from coordinators.neural_router import NeuralRouterWrapper
    coordinators["NeuralRouter"] = NeuralRouterWrapper(neural_router_coord)

    results = {}
    for name, coord in coordinators.items():
        agent_a = LocalCNNAgent(a_model, DEVICE)
        agent_b = LocalCNNAgent(b_model, DEVICE)
        system = MetaFlow(agents=[agent_a, agent_b], coordinator=coord)
        acc = evaluate_system(system, test_loader)
        results[name] = acc

    print("\n=== Coordinator Comparison (Teacher Accuracy on CIFAR-10 Test) ===\n")
    print(f"  Client A alone : {acc_a:.4%}")
    print(f"  Client B alone : {acc_b:.4%}")
    print()
    print(f"  {'Coordinator':<12} {'Teacher Acc':>12}  {'vs best client':>14}")
    print(f"  {'-'*42}")
    best_client = max(acc_a, acc_b)
    for name, acc in results.items():
        delta = acc - best_client
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<12} {acc:>12.4%}  {sign}{delta:>+.4%}")
    print()

    best_coord = max(results, key=results.get)
    print(f"  Best coordinator: {best_coord} ({results[best_coord]:.4%})")

    # extra stats for selection-based coordinators
    for name, coord in coordinators.items():
        if hasattr(coord, "stats") and coord.stats.get("total", 0) > 0:
            picked = coord.stats["picked"]
            total = coord.stats["total"]
            disagree_rate = coord.stats["disagree"] / total
            print(f"\n  [{name}] Agent picks: A={picked[0]} ({picked[0]/total:.1%}), "
                  f"B={picked[1]} ({picked[1]/total:.1%}) | Disagree rate: {disagree_rate:.1%}")


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
from metaflow.coordinators.confidence_select import ConfidenceSelectCoordinator
from metaflow.coordinators.margin_select import MarginSelectCoordinator

from metaflow.data import get_test_dataset

from metaflow.core import MetaFlow
from metaflow.coordinators.confidence_select import ConfidenceSelectCoordinator
from metaflow.agents.local_cnn_agent import LocalCNNAgent
from metaflow.evaluate import load_client_model, DEVICE


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


if __name__ == "__main__":
    test_loader = DataLoader(get_test_dataset(), batch_size=128, shuffle=False)

    a = load_client_model("a")
    b = load_client_model("b")

    agent_a = LocalCNNAgent(a, DEVICE)
    agent_b = LocalCNNAgent(b, DEVICE)

    coordinator = ConfidenceSelectCoordinator()

    system_a = MetaFlow(agents=[agent_a], coordinator=coordinator)
    system_b = MetaFlow(agents=[agent_b], coordinator=coordinator)
    system_ab = MetaFlow(agents=[agent_a, agent_b], coordinator=coordinator)

    acc_a = evaluate_system(system_a, test_loader)
    acc_b = evaluate_system(system_b, test_loader)
    acc_ab = evaluate_system(system_ab, test_loader)

    print(f"Agent A only accuracy: {acc_a:.4%}")
    print(f"Agent B only accuracy: {acc_b:.4%}")
    print(f"MetaFlow (A + B, {type(coordinator).__name__}) accuracy: {acc_ab:.4%}")
    print("Coordinator picks per agent:", coordinator.stats["picked"])
    for coordinator in [ConfidenceSelectCoordinator(), MarginSelectCoordinator()]:
        system_ab = MetaFlow(agents=[agent_a, agent_b], coordinator=coordinator)
        acc = evaluate_system(system_ab, test_loader)
        print(f"MetaFlow ({type(coordinator).__name__}) accuracy: {acc:.4%}")
        print("  picks:", coordinator.stats["picked"])
        print("  disagreement rate:", coordinator.stats["disagree"] / coordinator.stats["total"])

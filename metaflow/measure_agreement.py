"""
Measure agreement and disagreement patterns between two models.
Reports:
- A correct, B wrong
- A wrong, B correct
- Both correct
- Both wrong
- Router/coordinator accuracy (when router picks A/B, was it correct?)
"""
import torch
from torch.utils.data import DataLoader
from evaluate import load_client_model, DEVICE
from data import get_test_dataset
from core import MetaFlow
from agents.local_cnn_agent import LocalCNNAgent
from coordinators.confidence_select import ConfidenceSelectCoordinator
from coordinators.margin_select import MarginSelectCoordinator
from coordinators.agree_then_router import AgreeThenRouterCoordinator


def measure_model_agreement(model_a, model_b, loader):
    """
    Measure how often two models agree/disagree and their correctness patterns.
    
    Returns:
        dict with counts and percentages for each scenario
    """
    model_a.eval()
    model_b.eval()
    
    # Initialize counters
    both_correct = 0
    both_wrong = 0
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get predictions
            logits_a = model_a(x)
            logits_b = model_b(x)
            pred_a = logits_a.argmax(dim=-1)
            pred_b = logits_b.argmax(dim=-1)
            
            # Check correctness
            a_correct = pred_a == y
            b_correct = pred_b == y
            
            # Count scenarios
            both_correct += (a_correct & b_correct).sum().item()
            both_wrong += (~a_correct & ~b_correct).sum().item()
            a_correct_b_wrong += (a_correct & ~b_correct).sum().item()
            a_wrong_b_correct += (~a_correct & b_correct).sum().item()
            total += y.size(0)
    
    # Calculate percentages
    stats = {
        "total_samples": total,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_correct_pct": both_correct / total * 100,
        "both_wrong_pct": both_wrong / total * 100,
        "a_correct_b_wrong_pct": a_correct_b_wrong / total * 100,
        "a_wrong_b_correct_pct": a_wrong_b_correct / total * 100,
        "agree_count": both_correct + both_wrong,
        "disagree_count": a_correct_b_wrong + a_wrong_b_correct,
        "agreement_rate": (both_correct + both_wrong) / total * 100,
        "disagreement_rate": (a_correct_b_wrong + a_wrong_b_correct) / total * 100,
    }
    
    return stats


def measure_router_accuracy(model_a, model_b, coordinator, loader):
    """
    Measure how good the coordinator/router is at picking the correct model.
    
    Tracks:
    - When router picks A, how often is A correct?
    - When router picks B, how often is B correct?  
    - Overall router accuracy
    
    Returns:
        dict with router performance statistics
    """
    model_a.eval()
    model_b.eval()
    
    # Initialize counters
    picked_a_correct = 0
    picked_a_total = 0
    picked_b_correct = 0
    picked_b_total = 0
    total_correct = 0
    total_samples = 0
    
    # Get coordinator type to use appropriate selection logic
    coordinator_type = type(coordinator).__name__
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Ensure models are in eval mode
            model_a.eval()
            model_b.eval()
            
            # Get predictions from both models
            logits_a = model_a(x)
            logits_b = model_b(x)
            pred_a = logits_a.argmax(dim=-1)
            pred_b = logits_b.argmax(dim=-1)
            
            # Determine which model the coordinator would pick
            # Based on coordinator type
            if "Confidence" in coordinator_type:
                probs_a = torch.softmax(logits_a, dim=-1)
                probs_b = torch.softmax(logits_b, dim=-1)
                conf_a = probs_a.max(dim=-1).values
                conf_b = probs_b.max(dim=-1).values
                picked_a = conf_a >= conf_b
            elif "Margin" in coordinator_type:
                probs_a = torch.softmax(logits_a, dim=-1)
                probs_b = torch.softmax(logits_b, dim=-1)
                top2_a = probs_a.topk(k=2, dim=-1).values
                top2_b = probs_b.topk(k=2, dim=-1).values
                margin_a = top2_a[:, 0] - top2_a[:, 1]
                margin_b = top2_b[:, 0] - top2_b[:, 1]
                picked_a = margin_a >= margin_b
            else:
                # For other coordinators, run combine and compare
                combined_logits = coordinator.combine([logits_a, logits_b])
                final_pred = combined_logits.argmax(dim=-1)
                # Assume A picked if final matches A's pred and disagrees with B
                picked_a = (final_pred == pred_a) & (pred_a != pred_b)
                picked_a = picked_a | ((pred_a == pred_b) & (final_pred == pred_a))  # When they agree
                
            picked_b = ~picked_a
            
            # Check correctness
            a_correct = (pred_a == y)
            b_correct = (pred_b == y)
            
            # Track picks and correctness
            picked_a_total += picked_a.sum().item()
            picked_a_correct += (picked_a & a_correct).sum().item()
            
            picked_b_total += picked_b.sum().item()
            picked_b_correct += (picked_b & b_correct).sum().item()
            
            # Calculate router's actual performance
            router_pred = torch.where(picked_a, pred_a, pred_b)
            total_correct += (router_pred == y).sum().item()
            total_samples += y.size(0)
    
    # Calculate statistics
    stats = {
        "total_samples": total_samples,
        "picked_a_total": picked_a_total,
        "picked_b_total": picked_b_total,
        "picked_a_correct": picked_a_correct,
        "picked_b_correct": picked_b_correct,
        "picked_a_accuracy": picked_a_correct / picked_a_total * 100 if picked_a_total > 0 else 0,
        "picked_b_accuracy": picked_b_correct / picked_b_total * 100 if picked_b_total > 0 else 0,
        "router_accuracy": total_correct / total_samples * 100,
        "picked_a_pct": picked_a_total / total_samples * 100,
        "picked_b_pct": picked_b_total / total_samples * 100,
    }
    
    return stats


def print_router_accuracy_report(stats, coordinator_name="Router"):
    """Print a formatted report of router accuracy."""
    print("=" * 60)
    print(f"{coordinator_name.upper()} ACCURACY ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Overall router accuracy: {stats['router_accuracy']:.2f}%")
    print()
    
    print("WHEN ROUTER PICKS MODEL A:")
    print(f"  Times picked:    {stats['picked_a_total']:6d} ({stats['picked_a_pct']:6.2f}%)")
    print(f"  Correct picks:   {stats['picked_a_correct']:6d}")
    print(f"  Accuracy:        {stats['picked_a_accuracy']:6.2f}%")
    print()
    
    print("WHEN ROUTER PICKS MODEL B:")
    print(f"  Times picked:    {stats['picked_b_total']:6d} ({stats['picked_b_pct']:6.2f}%)")
    print(f"  Correct picks:   {stats['picked_b_correct']:6d}")
    print(f"  Accuracy:        {stats['picked_b_accuracy']:6.2f}%")
    print("=" * 60)



def print_agreement_report(stats):
    """Print a formatted report of agreement statistics."""
    print("=" * 60)
    print("MODEL AGREEMENT ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {stats['total_samples']}")
    print()
    
    print("AGREEMENT:")
    print(f"  Both correct:    {stats['both_correct']:6d} ({stats['both_correct_pct']:6.2f}%)")
    print(f"  Both wrong:      {stats['both_wrong']:6d} ({stats['both_wrong_pct']:6.2f}%)")
    print(f"  Total agree:     {stats['agree_count']:6d} ({stats['agreement_rate']:6.2f}%)")
    print()
    
    print("DISAGREEMENT:")
    print(f"  A correct, B wrong: {stats['a_correct_b_wrong']:6d} ({stats['a_correct_b_wrong_pct']:6.2f}%)")
    print(f"  A wrong, B correct: {stats['a_wrong_b_correct']:6d} ({stats['a_wrong_b_correct_pct']:6.2f}%)")
    print(f"  Total disagree:     {stats['disagree_count']:6d} ({stats['disagreement_rate']:6.2f}%)")
    print()
    
    # Additional insights
    if stats['disagree_count'] > 0:
        a_better = stats['a_correct_b_wrong'] / stats['disagree_count'] * 100
        b_better = stats['a_wrong_b_correct'] / stats['disagree_count'] * 100
        print("WHEN THEY DISAGREE:")
        print(f"  A is correct:    {a_better:6.2f}%")
        print(f"  B is correct:    {b_better:6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    # Load models
    print("Loading models...")
    client_a = load_client_model("a")
    client_b = load_client_model("b")
    
    # Load test data
    test_ds = get_test_dataset()
    
    # Measure basic agreement
    print("Measuring agreement between Client A and Client B...")
    print()
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    stats = measure_model_agreement(client_a, client_b, test_loader)
    print_agreement_report(stats)
    print()
    
    # Measure router accuracy for different coordinators
    coordinators = [
        ("Confidence", ConfidenceSelectCoordinator()),
        ("Margin", MarginSelectCoordinator()),
    ]
    
    # Try to add AgreeThenRouter if it's available
    try:
        from data import get_probe_splits
        router_probe_ds, _ = get_probe_splits(seed=42)
        router_coord = AgreeThenRouterCoordinator()
        router_coord.fit_from_models(client_a, client_b, router_probe_ds, DEVICE, seed=42)
        if router_coord.is_trained and router_coord.stats.get("router_enabled", False):
            coordinators.append(("AgreeRouter", router_coord))
    except Exception as e:
        print(f"Note: Could not load AgreeRouter coordinator: {e}")
    
    for coord_name, coordinator in coordinators:
        print()
        # Create fresh loader for each coordinator
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        router_stats = measure_router_accuracy(client_a, client_b, coordinator, test_loader)
        print_router_accuracy_report(router_stats, coord_name)


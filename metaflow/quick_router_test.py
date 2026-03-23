"""
Quick test of router accuracy on a subset of test data.
"""
import torch
from torch.utils.data import DataLoader, Subset
from evaluate import load_client_model, DEVICE
from data import get_test_dataset
from coordinators.confidence_select import ConfidenceSelectCoordinator
from coordinators.margin_select import MarginSelectCoordinator


def quick_router_accuracy(model_a, model_b, coordinator, loader, coordinator_name=""):
    """Quick measurement of router accuracy."""
    model_a.eval()
    model_b.eval()
    
    picked_a_correct = 0
    picked_a_total = 0
    picked_b_correct = 0
    picked_b_total = 0
    total_correct = 0
    total_samples = 0
    
    coordinator_type = type(coordinator).__name__
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get predictions
            logits_a = model_a(x)
            logits_b = model_b(x)
            pred_a = logits_a.argmax(dim=-1)
            pred_b = logits_b.argmax(dim=-1)
            
            # Determine picks based on coordinator type
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
                picked_a = torch.ones(y.size(0), dtype=torch.bool, device=DEVICE)
                
            picked_b = ~picked_a
            
            # Check correctness
            a_correct = (pred_a == y)
            b_correct = (pred_b == y)
            
            # Track statistics
            picked_a_total += picked_a.sum().item()
            picked_a_correct += (picked_a & a_correct).sum().item()
            
            picked_b_total += picked_b.sum().item()
            picked_b_correct += (picked_b & b_correct).sum().item()
            
            # Router's actual performance
            router_pred = torch.where(picked_a, pred_a, pred_b)
            total_correct += (router_pred == y).sum().item()
            total_samples += y.size(0)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{coordinator_name.upper()} ROUTER ACCURACY")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Overall router accuracy: {total_correct / total_samples * 100:.2f}%")
    print()
    print(f"When router picks Model A (n={picked_a_total}):")
    print(f"  Accuracy: {picked_a_correct / picked_a_total * 100:.2f}%" if picked_a_total > 0 else "  N/A")
    print()
    print(f"When router picks Model B (n={picked_b_total}):")
    print(f"  Accuracy: {picked_b_correct / picked_b_total * 100:.2f}%" if picked_b_total > 0 else "  N/A")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Loading models...")
    client_a = load_client_model("a")
    client_b = load_client_model("b")
    
    # Use subset for faster testing
    print("Loading test data (using 2000 samples for speed)...")
    test_ds = get_test_dataset()
    subset_indices = list(range(2000))  # First 2000 samples
    test_subset = Subset(test_ds, subset_indices)
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    
    # Test Confidence coordinator
    print("\nTesting Confidence coordinator...")
    conf_coord = ConfidenceSelectCoordinator()
    quick_router_accuracy(client_a, client_b, conf_coord, test_loader, "Confidence")
    
    # Test Margin coordinator
    print("Testing Margin coordinator...")
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)  # Fresh loader
    margin_coord = MarginSelectCoordinator()
    quick_router_accuracy(client_a, client_b, margin_coord, test_loader, "Margin")
    
    print("\nDone! For full test on all 10,000 samples, use measure_agreement.py")

"""
Quick test of router accuracy on a subset of test data.
Tests all available coordinators/routers.
"""
import torch
from torch.utils.data import DataLoader, Subset
from evaluate import load_client_model, DEVICE
from data import get_test_dataset, get_probe_splits
from coordinators.confidence_select import ConfidenceSelectCoordinator
from coordinators.margin_select import MarginSelectCoordinator
from coordinators.agree_then_router import AgreeThenRouterCoordinator
from coordinators.EnsembleCoordinator import EnsembleCoordinator
from coordinators.average_logits import AverageLogitsCoordinator


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
    
    # Disagreement-only metrics
    disagree_total = 0
    disagree_correct = 0
    
    coordinator_type = type(coordinator).__name__
    is_selector = "Confidence" in coordinator_type or "Margin" in coordinator_type or "Agree" in coordinator_type
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get predictions
            logits_a = model_a(x)
            logits_b = model_b(x)
            pred_a = logits_a.argmax(dim=-1)
            pred_b = logits_b.argmax(dim=-1)
            
            # Get combined output from coordinator
            combined_logits = coordinator.combine([logits_a, logits_b])
            final_pred = combined_logits.argmax(dim=-1)
            
            # Track disagreement cases
            disagree_mask = pred_a != pred_b
            disagree_total += disagree_mask.sum().item()
            disagree_correct += (disagree_mask & (final_pred == y)).sum().item()
            
            # For selector coordinators, determine which model was picked
            if is_selector:
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
                elif "Agree" in coordinator_type:
                    # When models agree, both are "picked"; when they disagree, check which one's prediction matches final
                    agree = pred_a == pred_b
                    picked_a = agree | (final_pred == pred_a)
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
            
            # Overall accuracy
            total_correct += (final_pred == y).sum().item()
            total_samples += y.size(0)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{coordinator_name.upper()} ROUTER ACCURACY")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {total_correct / total_samples * 100:.2f}%")
    
    if is_selector and picked_a_total + picked_b_total > 0:
        print()
        print(f"When router picks Model A (n={picked_a_total}):")
        if picked_a_total > 0:
            print(f"  Accuracy: {picked_a_correct / picked_a_total * 100:.2f}%")
        else:
            print(f"  N/A")
        print()
        print(f"When router picks Model B (n={picked_b_total}):")
        if picked_b_total > 0:
            print(f"  Accuracy: {picked_b_correct / picked_b_total * 100:.2f}%")
        else:
            print(f"  N/A")
    else:
        print("(Blending coordinator - combines both models' outputs)")
    
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
    
    print("\n" + "="*60)
    print("TESTING ALL COORDINATORS")
    print("="*60)
    
    # Test Confidence coordinator
    print("\n[1/5] Testing Confidence coordinator...")
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    conf_coord = ConfidenceSelectCoordinator()
    quick_router_accuracy(client_a, client_b, conf_coord, test_loader, "Confidence")
    
    # Test Margin coordinator
    print("[2/5] Testing Margin coordinator...")
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    margin_coord = MarginSelectCoordinator()
    quick_router_accuracy(client_a, client_b, margin_coord, test_loader, "Margin")
    
    # Test Ensemble coordinator (default)
    print("[3/5] Testing Ensemble coordinator (default)...")
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    ensemble_coord = EnsembleCoordinator()
    quick_router_accuracy(client_a, client_b, ensemble_coord, test_loader, "Ensemble")
    
    # Test AverageLogits coordinator
    print("[4/5] Testing AverageLogits coordinator...")
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
    avg_coord = AverageLogitsCoordinator()
    quick_router_accuracy(client_a, client_b, avg_coord, test_loader, "AverageLogits")
    
    # Test AgreeThenRouter coordinator (needs training)
    print("[5/5] Testing AgreeThenRouter coordinator...")
    try:
        router_probe_ds, _ = get_probe_splits(seed=42)
        router_coord = AgreeThenRouterCoordinator()
        print("  Training router on probe data...")
        router_coord.fit_from_models(client_a, client_b, router_probe_ds, DEVICE, seed=42)
        
        if router_coord.is_trained and router_coord.stats.get("router_enabled", False):
            print(f"  Router trained: samples={router_coord.stats['router_train_samples']}, "
                  f"val_acc={router_coord.stats.get('router_val_acc', 0):.2%}")
            test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
            quick_router_accuracy(client_a, client_b, router_coord, test_loader, "AgreeThenRouter")
        else:
            print(f"  Router not enabled (insufficient training data)")
    except Exception as e:
        print(f"  Could not test AgreeThenRouter: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Tested 5 coordinators:")
    print("  - Confidence: Picks model with highest confidence")
    print("  - Margin: Picks model with highest margin (top1 - top2)")
    print("  - Ensemble: Averages probabilities (default)")
    print("  - AverageLogits: Averages logits directly")
    print("  - AgreeThenRouter: Uses trained NN when models disagree")
    print("="*60)
    
    print("\nDone! For full test on all 10,000 samples, use measure_agreement.py")

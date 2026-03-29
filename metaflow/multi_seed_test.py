"""
Multi-seed robustness test for router accuracy.
Tests whether coordinator performance is consistent across different random seeds.

Reports mean ± std for:
- Expert A accuracy
- Expert B accuracy
- Router/coordinator accuracy
- Disagreement-only accuracy
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from evaluate import load_client_model, DEVICE
from data import get_test_dataset, get_probe_splits
from coordinators.confidence_select import ConfidenceSelectCoordinator
from coordinators.margin_select import MarginSelectCoordinator
from coordinators.agree_then_router import AgreeThenRouterCoordinator
from coordinators.EnsembleCoordinator import EnsembleCoordinator


def measure_accuracy_single_seed(model_a, model_b, coordinator, loader, coordinator_name=""):
    """Measure accuracy metrics for a single seed."""
    model_a.eval()
    model_b.eval()
    
    # Overall metrics
    a_correct = 0
    b_correct = 0
    router_correct = 0
    total_samples = 0
    
    # Disagreement-only metrics
    disagree_total = 0
    disagree_correct = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Get predictions
            logits_a = model_a(x)
            logits_b = model_b(x)
            pred_a = logits_a.argmax(dim=-1)
            pred_b = logits_b.argmax(dim=-1)
            
            # Router prediction
            combined_logits = coordinator.combine([logits_a, logits_b])
            final_pred = combined_logits.argmax(dim=-1)
            
            # Track overall correctness
            a_correct += (pred_a == y).sum().item()
            b_correct += (pred_b == y).sum().item()
            router_correct += (final_pred == y).sum().item()
            total_samples += y.size(0)
            
            # Track disagreement cases
            disagree_mask = pred_a != pred_b
            disagree_total += disagree_mask.sum().item()
            disagree_correct += (disagree_mask & (final_pred == y)).sum().item()
    
    return {
        'expert_a_acc': a_correct / total_samples * 100,
        'expert_b_acc': b_correct / total_samples * 100,
        'router_acc': router_correct / total_samples * 100,
        'disagree_rate': disagree_total / total_samples * 100,
        'disagree_acc': disagree_correct / disagree_total * 100 if disagree_total > 0 else 0,
    }


def test_coordinator_across_seeds(coordinator_name, client_a, client_b, test_subset, num_seeds=10):
    """Test a coordinator across multiple seeds (models loaded once)."""
    print(f"\n{'='*70}")
    print(f"TESTING {coordinator_name.upper()} ACROSS {num_seeds} SEEDS")
    print(f"{'='*70}")
    
    results = []
    
    for seed in range(num_seeds):
        print(f"Seed {seed:2d}... ", end='', flush=True)
        
        # Set seed for reproducibility (matches train_local.py pattern)
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Create coordinator
        if coordinator_name == "Confidence":
            coordinator = ConfidenceSelectCoordinator()
        elif coordinator_name == "Margin":
            coordinator = MarginSelectCoordinator()
        elif coordinator_name == "Ensemble":
            coordinator = EnsembleCoordinator()
        elif coordinator_name == "AgreeThenRouter":
            coordinator = AgreeThenRouterCoordinator()
            try:
                print("training... ", end='', flush=True)
                router_probe_ds, _ = get_probe_splits(seed=seed)
                coordinator.fit_from_models(client_a, client_b, router_probe_ds, DEVICE, seed=seed)
                if not (coordinator.is_trained and coordinator.stats.get("router_enabled", False)):
                    print(f"SKIP (router not enabled)")
                    continue
            except Exception as e:
                print(f"SKIP ({str(e)[:50]})")
                continue
        else:
            print(f"SKIP (unknown coordinator)")
            continue
        
        # Measure accuracy
        test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)
        metrics = measure_accuracy_single_seed(client_a, client_b, coordinator, test_loader, coordinator_name)
        results.append(metrics)
        
        print(f"Router: {metrics['router_acc']:.2f}%, Disagree: {metrics['disagree_acc']:.2f}%")
    
    if not results:
        print("\nNo valid results collected!")
        return None
    
    # Calculate statistics
    expert_a_accs = [r['expert_a_acc'] for r in results]
    expert_b_accs = [r['expert_b_acc'] for r in results]
    router_accs = [r['router_acc'] for r in results]
    disagree_rates = [r['disagree_rate'] for r in results]
    disagree_accs = [r['disagree_acc'] for r in results]
    
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {coordinator_name.upper()} ({len(results)} seeds)")
    print(f"{'='*70}")
    print(f"Expert A accuracy:      {np.mean(expert_a_accs):6.2f}% ± {np.std(expert_a_accs):5.2f}%")
    print(f"Expert B accuracy:      {np.mean(expert_b_accs):6.2f}% ± {np.std(expert_b_accs):5.2f}%")
    print(f"Router accuracy:        {np.mean(router_accs):6.2f}% ± {np.std(router_accs):5.2f}%")
    print(f"Disagreement rate:      {np.mean(disagree_rates):6.2f}% ± {np.std(disagree_rates):5.2f}%")
    print(f"Disagreement-only acc:  {np.mean(disagree_accs):6.2f}% ± {np.std(disagree_accs):5.2f}%")
    print(f"{'='*70}")
    
    return {
        'coordinator': coordinator_name,
        'num_seeds': len(results),
        'expert_a': (np.mean(expert_a_accs), np.std(expert_a_accs)),
        'expert_b': (np.mean(expert_b_accs), np.std(expert_b_accs)),
        'router': (np.mean(router_accs), np.std(router_accs)),
        'disagree_rate': (np.mean(disagree_rates), np.std(disagree_rates)),
        'disagree_acc': (np.mean(disagree_accs), np.std(disagree_accs)),
    }


if __name__ == "__main__":
    print("="*70)
    print("MULTI-SEED ROBUSTNESS TEST")
    print("="*70)
    print("Testing coordinator performance across multiple random seeds")
    print("to ensure results are not due to lucky initialization.")
    print()
    print("NOTE: Deterministic coordinators (Confidence, Margin, Ensemble)")
    print("      will show std=0 since they produce identical results")
    print("      with fixed models and test data. Only AgreeThenRouter")
    print("      has stochastic training, so only it will show variance.")
    print()
    
    NUM_SEEDS = 5  # Reduced for speed
    NUM_SAMPLES = 2000
    
    print(f"Configuration:")
    print(f"  - Number of seeds: {NUM_SEEDS}")
    print(f"  - Test samples per seed: {NUM_SAMPLES}")
    print(f"  - Device: {DEVICE}")
    
    # Load models and test data ONCE (they don't change with seed)
    print("\nLoading models and test data...")
    client_a = load_client_model("a")
    client_b = load_client_model("b")
    test_ds = get_test_dataset()
    subset_indices = list(range(NUM_SAMPLES))
    test_subset = Subset(test_ds, subset_indices)
    print("  Models loaded successfully")
    
    # Test all coordinators
    coordinators_to_test = [
        "Confidence",
        "Margin",
        "Ensemble",
        "AgreeThenRouter",
    ]
    
    all_results = {}
    
    for coord_name in coordinators_to_test:
        result = test_coordinator_across_seeds(coord_name, client_a, client_b, test_subset, num_seeds=NUM_SEEDS)
        if result:
            all_results[coord_name] = result
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL COORDINATORS")
    print("="*70)
    print(f"{'Coordinator':<20} {'Router Acc':<20} {'Disagree-Only Acc':<20}")
    print("-"*70)
    
    for coord_name in coordinators_to_test:
        if coord_name in all_results:
            r = all_results[coord_name]
            router_mean, router_std = r['router']
            disagree_mean, disagree_std = r['disagree_acc']
            print(f"{coord_name:<20} {router_mean:5.2f}% ± {router_std:4.2f}%     "
                  f"{disagree_mean:5.2f}% ± {disagree_std:4.2f}%")
    
    print("="*70)
    print("\nKey Question: Does the AgreeThenRouter advantage hold across seeds?")
    
    if "AgreeThenRouter" in all_results and "Confidence" in all_results:
        router_mean = all_results["AgreeThenRouter"]["disagree_acc"][0]
        conf_mean = all_results["Confidence"]["disagree_acc"][0]
        advantage = router_mean - conf_mean
        print(f"AgreeThenRouter vs Confidence (disagreement-only): +{advantage:.2f} pp")
    
    print("\nDone!")

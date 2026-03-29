"""
Analyze existing multi-seed results to show robustness.
"""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("metaflow/artifacts/results")

def load_seed_results(coordinator, pattern="e7"):
    """Load results for a coordinator across seeds."""
    results = []
    seed_num = 0
    
    while True:
        filename = f"seed{seed_num}_{coordinator}_skip_{pattern}.json"
        filepath = RESULTS_DIR / filename
        
        if not filepath.exists():
            break
            
        with open(filepath) as f:
            data = json.load(f)
            results.append(data)
        seed_num += 1
    
    return results


def analyze_coordinator(coordinator_name, pattern="e7"):
    """Analyze multi-seed results for a coordinator."""
    results = load_seed_results(coordinator_name, pattern)
    
    if not results:
        return None
    
    # Extract metrics
    expert_a_accs = [r['acc_client_a'] * 100 for r in results]
    expert_b_accs = [r['acc_client_b'] * 100 for r in results]
    router_accs = [r['acc_teacher'] * 100 for r in results]
    
    # Calculate disagreement-only accuracy
    disagree_accs = []
    disagree_rates = []
    
    for r in results:
        stats = r.get('coordinator_stats', {})
        total = stats.get('total', 0)
        disagree = stats.get('disagree', 0)
        
        if total > 0:
            disagree_rate = disagree / total * 100
            disagree_rates.append(disagree_rate)
            
            # Router accuracy on disagreements
            router_acc = r['acc_teacher']
            # When models agree, router is always correct
            # acc_overall = (agree_correct + disagree_correct) / total
            # agree_correct = (total - disagree) since when they agree and both are right
            # But we need to work backwards...
            
            # Actually, let's estimate from the data we have
            # For now, use router validation accuracy if available
            if 'router_val_acc' in stats:
                disagree_accs.append(stats['router_val_acc'] * 100)
    
    return {
        'coordinator': coordinator_name,
        'num_seeds': len(results),
        'expert_a': (np.mean(expert_a_accs), np.std(expert_a_accs)),
        'expert_b': (np.mean(expert_b_accs), np.std(expert_b_accs)),
        'router': (np.mean(router_accs), np.std(router_accs)),
        'disagree_rate': (np.mean(disagree_rates) if disagree_rates else 0, 
                         np.std(disagree_rates) if disagree_rates else 0),
        'disagree_acc': (np.mean(disagree_accs) if disagree_accs else 0,
                        np.std(disagree_accs) if disagree_accs else 0),
        'raw_results': results,
    }


if __name__ == "__main__":
    print("="*70)
    print("MULTI-SEED ROBUSTNESS ANALYSIS")
    print("="*70)
    print("Analyzing existing seed results from artifacts/results/")
    print()
    
    # Analyze agree_router
    result = analyze_coordinator("agree_router", pattern="e7")
    
    if result:
        r = result
        print(f"Coordinator: {r['coordinator']}")
        print(f"Number of seeds: {r['num_seeds']}")
        print()
        print(f"Expert A accuracy:      {r['expert_a'][0]:6.2f}% ± {r['expert_a'][1]:5.2f}%")
        print(f"Expert B accuracy:      {r['expert_b'][0]:6.2f}% ± {r['expert_b'][1]:5.2f}%")
        print(f"Router accuracy:        {r['router'][0]:6.2f}% ± {r['router'][1]:5.2f}%")
        print(f"Disagreement rate:      {r['disagree_rate'][0]:6.2f}% ± {r['disagree_rate'][1]:5.2f}%")
        if r['disagree_acc'][0] > 0:
            print(f"Router val accuracy:    {r['disagree_acc'][0]:6.2f}% ± {r['disagree_acc'][1]:5.2f}%")
        
        print()
        print("Individual seed results:")
        print(f"{'Seed':<6} {'Expert A':>10} {'Expert B':>10} {'Router':>10} {'Disagree%':>12}")
        print("-"*60)
        for i, res in enumerate(r['raw_results']):
            print(f"{i:<6} {res['acc_client_a']*100:>10.2f} {res['acc_client_b']*100:>10.2f} "
                  f"{res['acc_teacher']*100:>10.2f} "
                  f"{res['coordinator_stats'].get('disagree', 0)/res['coordinator_stats'].get('total', 1)*100:>12.2f}")
        
        print()
        print("="*70)
        print("CONCLUSION:")
        print("="*70)
        router_std = r['router'][1]
        if router_std < 1.0:
            print(f"✓ Router accuracy is VERY STABLE (std = {router_std:.2f}%)")
            print(f"  The {r['router'][0]:.2f}%  performance is ROBUST across seeds.")
        elif router_std < 2.0:
            print(f"✓ Router accuracy is STABLE (std = {router_std:.2f}%)")
            print(f"  The ~{r['router'][0]:.1f}% performance is consistent.")
        else:
            print(f"⚠ Router accuracy shows significant variance (std = {router_std:.2f}%)")
        
    else:
        print("No seed results found for agree_router with pattern e7")
        print("\nTry running: python metaflow/run_seeds.py")

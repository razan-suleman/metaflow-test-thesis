# metaflow/ensemble.py
import os
import torch
import torch.nn.functional as F

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def build_ensemble(client_names=None, weights=None):
    """
    Load probe logits for each client, turn to probabilities, and build ensemble.
    Saves a tensor of shape (N_probes, num_classes).
    """
    if client_names is None:
        client_names = ["client_a", "client_b"]

    num_clients = len(client_names)

    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1."

    client_probs = []
    for name in client_names:
        path = os.path.join(CHECKPOINT_DIR, f"{name}_probe_logits.pt")
        logits = torch.load(path, map_location="cpu")
        probs = F.softmax(logits, dim=1)  # (N_probes, num_classes)
        client_probs.append(probs)

    # Check they all have same shape
    num_samples = client_probs[0].shape[0]
    num_classes = client_probs[0].shape[1]

    ensemble = torch.zeros(num_samples, num_classes)
    for w, p in zip(weights, client_probs):
        ensemble += w * p

    out_path = os.path.join(CHECKPOINT_DIR, "ensemble_probe_probs.pt")
    torch.save(ensemble, out_path)
    print(f"Saved ensemble probe probabilities to {out_path}")


if __name__ == "__main__":
    build_ensemble()

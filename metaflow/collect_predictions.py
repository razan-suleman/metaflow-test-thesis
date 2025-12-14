import os
import torch
from torch.utils.data import DataLoader

from data import get_probe_dataset
from models.local_model import LocalCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
PREDICTIONS_DIR = "predictions"


def load_client_model(client: str) -> LocalCNN:
    model = LocalCNN().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"client_{client}.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_for_client(client: str, batch_size: int = 128):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    probe_ds = get_probe_dataset()
    loader = DataLoader(probe_ds, batch_size=batch_size, shuffle=False)

    model = load_client_model(client)

    all_logits = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            logits = model(x)  # shape [B, num_classes]
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    torch.save(
        {
            "logits": all_logits,
            "num_samples": len(probe_ds),
        },
        os.path.join(PREDICTIONS_DIR, f"client_{client}_probes.pt"),
    )
    print(f"Saved probe logits for client {client}.")


if __name__ == "__main__":
    collect_for_client("a")
    collect_for_client("b")

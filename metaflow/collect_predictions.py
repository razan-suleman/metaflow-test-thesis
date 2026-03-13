import os
import torch
from torch.utils.data import DataLoader

from data import get_probe_dataset
from models.local_model import LocalCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
# revised: use artifacts folder for probe logits instead of predictions/
ARTIFACTS_DIR = "artifacts/probe_logits"


def load_client_model(client: str) -> LocalCNN:
    model = LocalCNN().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"client_{client}.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def collect_for_client(client: str, batch_size: int = 128):
    # ensure output directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

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
    out_path = os.path.join(ARTIFACTS_DIR, f"client_{client}_probe_logits.pt")
    torch.save(
        {
            "logits": all_logits,
            "num_samples": len(probe_ds),
        },
        out_path,
    )
    print(f"Saved probe logits for client {client} at {out_path}.")


if __name__ == "__main__":
    collect_for_client("a")
    collect_for_client("b")

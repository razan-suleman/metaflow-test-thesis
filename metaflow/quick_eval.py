import torch
from torch.utils.data import DataLoader
from models.local_model import LocalCNN
from data import get_test_dataset

print("Loading test dataset...")
test_data = get_test_dataset()
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
print(f"Test dataset size: {len(test_data)}")

print("\nLoading Client A model...")
model_a = LocalCNN()
model_a.load_state_dict(torch.load("checkpoints/client_a.pt", map_location="cpu"))
model_a.eval()

print("Evaluating Client A...")
correct, total = 0, 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        if i % 10 == 0:
            print(f"  Batch {i}/{len(test_loader)}", end="\r")
        logits = model_a(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)

acc_a = correct / total
print(f"\n✓ Client A Test Accuracy: {acc_a * 100:.2f}%")

print("\nLoading Client B model...")
model_b = LocalCNN()
model_b.load_state_dict(torch.load("checkpoints/client_b.pt", map_location="cpu"))
model_b.eval()

print("Evaluating Client B...")
correct, total = 0, 0
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        if i % 10 == 0:
            print(f"  Batch {i}/{len(test_loader)}", end="\r")
        logits = model_b(x)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)

acc_b = correct / total
print(f"\n✓ Client B Test Accuracy: {acc_b * 100:.2f}%")

print(f"\n===== RESULTS =====")
print(f"Client A: {acc_a * 100:.2f}%")
print(f"Client B: {acc_b * 100:.2f}%")
print(f"Best: {max(acc_a, acc_b) * 100:.2f}%")

import torch
from models.local_model import LocalCNN

print("Loading checkpoint...")
checkpoint = torch.load("checkpoints/client_a.pt", map_location="cpu")
print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")

print("\nCreating model...")
model = LocalCNN()
print("Model state dict keys:")
for key, param in model.state_dict().items():
    print(f"  {key}: {param.shape}")

print("\nCheckpoint state dict keys:")
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
else:
    print("Checkpoint format unexpected")
    state_dict = None

if state_dict:
    for key, param in state_dict.items():
        print(f"  {key}: {param.shape}")
    
    print("\nLoading checkpoint into model...")
    try:
        model.load_state_dict(state_dict)
        print("✓ Successfully loaded checkpoint!")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")

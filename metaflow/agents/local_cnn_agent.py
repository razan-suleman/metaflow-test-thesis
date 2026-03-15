import torch
from models.local_model import LocalCNN

class LocalCNNAgent:
    def __init__(self, model: LocalCNN, device: str):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))

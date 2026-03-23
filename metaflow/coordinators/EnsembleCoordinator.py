import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class EnsembleCoordinator:
    def combine(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        return torch.log(avg_probs + 1e-8)
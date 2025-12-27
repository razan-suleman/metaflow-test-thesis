from typing import List
import torch

class AverageLogitsCoordinator:
    def combine(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        # logits_list: [B,10] each
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)

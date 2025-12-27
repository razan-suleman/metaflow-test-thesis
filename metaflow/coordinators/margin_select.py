from typing import List
import torch

class MarginSelectCoordinator:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.stats = {"picked": [], "total": 0, "disagree": 0}

    def combine(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        if len(logits_list) == 0:
            raise ValueError("logits_list is empty")
        if len(logits_list) == 1:
            return logits_list[0]

        stacked = torch.stack(logits_list, dim=0)
        probs = torch.softmax(stacked / self.temperature, dim=-1)

        top2 = probs.topk(k=2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        best_idx = margin.argmax(dim=0)

        n_agents = stacked.size(0)
        if (not self.stats["picked"]) or (len(self.stats["picked"]) != n_agents):
            self.stats["picked"] = [0] * n_agents
        for i in range(n_agents):
            self.stats["picked"][i] += (best_idx == i).sum().item()

        preds = stacked.argmax(dim=-1)
        self.stats["total"] += preds.size(1)
        all_same = (preds == preds[0:1]).all(dim=0)
        self.stats["disagree"] += (~all_same).sum().item()

        gather_idx = best_idx.view(1, -1, 1).expand(1, stacked.size(1), stacked.size(2))
        return torch.gather(stacked, dim=0, index=gather_idx).squeeze(0)

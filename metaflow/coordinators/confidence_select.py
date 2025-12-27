from typing import List
import torch

class ConfidenceSelectCoordinator:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.stats = {
        "picked": [],
        "disagree": 0,
        "total": 0,
    }

    def combine(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        if len(logits_list) == 0:
            raise ValueError("logits_list is empty")
        if len(logits_list) == 1:
            return logits_list[0]

        stacked = torch.stack(logits_list, dim=0)                 # [N,B,C]
        probs = torch.softmax(stacked / self.temperature, dim=-1) # [N,B,C]
        conf = probs.max(dim=-1).values                           # [N,B]
        best_idx = conf.argmax(dim=0)                             # [B]

        # stats: picked
        n_agents = stacked.size(0)
        if (not self.stats["picked"]) or (len(self.stats["picked"]) != n_agents):
            self.stats["picked"] = [0] * n_agents
        for i in range(n_agents):
            self.stats["picked"][i] += (best_idx == i).sum().item()

        # stats: disagreement (N-agent)
        preds = stacked.argmax(dim=-1)                            # [N,B]
        self.stats["total"] += preds.size(1)
        all_same = (preds == preds[0:1]).all(dim=0)               # [B]
        self.stats["disagree"] += (~all_same).sum().item()

        gather_idx = best_idx.view(1, -1, 1).expand(1, stacked.size(1), stacked.size(2))
        chosen = torch.gather(stacked, dim=0, index=gather_idx).squeeze(0)  # [B,C]
        return chosen

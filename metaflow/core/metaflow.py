from dataclasses import dataclass
from typing import List, Protocol
import torch

class Agent(Protocol):
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor: ...

class Coordinator(Protocol):
    def combine(self, logits_list: List[torch.Tensor]) -> torch.Tensor: ...

@dataclass
class MetaFlow:
    agents: List[Agent]
    coordinator: Coordinator

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits_list = [agent.predict_logits(x) for agent in self.agents]
        return self.coordinator.combine(logits_list)

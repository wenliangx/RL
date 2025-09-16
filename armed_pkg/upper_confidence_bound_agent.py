import numpy as np
import sys
sys.path.append('.')
from .__agent_base__ import __AgentBase__
from .upper_confidence_bound_action_selection import upperConfidenceBoundActionSelection, upperConfidenceBoundActionSelectionTorch
class UpperConfidenceBoundAgent(__AgentBase__):
    def __init__(self, k=10, initial=0, c=2):
        super().__init__(k=k, initial=initial)
        self.c = c

    def select_action(self, **kwargs) -> int:
        return upperConfidenceBoundActionSelection(self.q_estimation, self.action_count, np.sum(self.action_count), c=kwargs.get('c', self.c))

    def update_estimation(self, action, reward, **kwargs) -> None:
        self.action_count[action] += 1
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]


import torch
from .__agent_base__ import __AgentBaseTorch__
class UpperConfidenceBoundAgentTorch(__AgentBaseTorch__):
    def __init__(self, k=10, env_num=10, initial=0, c=2, device='cuda'):
        super().__init__(k=k, env_num=env_num, initial=initial, device=device)
        self.c = c

    def select_action(self, **kwargs) -> torch.Tensor:
        total_count = torch.sum(self.action_count, dim=1).unsqueeze(1).repeat(1, self.k)
        return upperConfidenceBoundActionSelectionTorch(self.q_estimation, self.action_count, total_count, c=kwargs.get('c', self.c))

    def update_estimation(self, action: torch.Tensor, reward: torch.Tensor, **kwargs) -> None:
        super().update_estimation(action, reward, **kwargs)
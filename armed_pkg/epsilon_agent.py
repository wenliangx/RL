import numpy as np
import sys
sys.path.append('.')
from .__agent_base__ import __AgentBase__
from .epsilon_greedy_action_selection import epsilonGreedyActionSelection

class EpsilonAgent(__AgentBase__):
    def __init__(self, k=10, initial=0, epsilon=0.1):
        super().__init__(k=k, initial=initial)
        self.epsilon = epsilon

    def select_action(self, **kwargs) -> int:
        return epsilonGreedyActionSelection(self.q_estimation, epsilon=self.epsilon)

    def update_estimation(self, action, reward, **kwargs) -> None:
        super().update_estimation(action, reward, **kwargs)

import torch
from .__agent_base__ import __AgentBaseTorch__
from .epsilon_greedy_action_selection import epsilonGreedyActionSelectionTorch

class EpsilonAgentTorch(__AgentBaseTorch__):
    def __init__(self, k=10, env_num=10, initial=0, epsilon=0.1, device='cuda'):
        super().__init__(k=k, env_num=env_num, initial=initial, device=device)
        self.epsilon = epsilon

    def select_action(self, **kwargs) -> torch.Tensor:
        return epsilonGreedyActionSelectionTorch(self.q_estimation, epsilon=self.epsilon)

    def update_estimation(self, action: torch.Tensor, reward: torch.Tensor, **kwargs) -> None:
        super().update_estimation(action, reward, **kwargs)
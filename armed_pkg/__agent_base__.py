import numpy as np
from abc import ABC, abstractmethod
import torch

class __AgentBase__(ABC):
    def __init__(self, k=10, initial=0):
        self.k = k
        self.initial = initial
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)

    @abstractmethod
    def select_action(self, **kwargs) -> int:
        pass

    def update_estimation(self, action, reward, **kwargs) -> None:
        self.action_count[action] += 1
        alpha = kwargs.get('alpha', None)
        if alpha is not None:
            self.q_estimation[action] += alpha * (reward - self.q_estimation[action])
        else:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]

class __AgentBaseTorch__(ABC):
    def __init__(self, k=10, num_env=10, initial=0, device='cuda'):
        self.k = k
        self.num_env = num_env
        self.initial = initial
        self.device = device
        self.q_estimation = torch.full(
            (self.num_env, self.k), self.initial, device=self.device, dtype=torch.float
        )
        self.action_count = torch.zeros((self.num_env, self.k), device=self.device)

    @abstractmethod
    def select_action(self, **kwargs) -> torch.Tensor:
        pass

    def update_estimation(self, action: torch.Tensor, reward: torch.Tensor, **kwargs) -> None:
        if action.dtype != torch.int64:
            raise ValueError("Action tensor must be of type torch.int64")
        
        action_indices = torch.arange(self.num_env, device=self.device)
        self.action_count[action_indices, action] += 1
        alpha = kwargs.get('alpha', None)
        if alpha is not None:
            self.q_estimation[action_indices, action] += alpha * (reward[action_indices] - self.q_estimation[action_indices, action])
        else:
            self.q_estimation[action_indices, action] += (reward[action_indices] - self.q_estimation[action_indices, action]) / self.action_count[action_indices, action]
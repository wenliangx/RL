import numpy as np
from abc import ABC, abstractmethod
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
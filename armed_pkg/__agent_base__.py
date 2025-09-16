import numpy as np
from abc import ABC, abstractmethod
class __AgentBase__(ABC):
    def __init__(self, k=10, initial=0):
        self.k = k
        self.initial = initial
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)

    @abstractmethod
    def select_action(self, **kwargs) -> np.intp:
        pass

    @abstractmethod
    def update_estimation(self, action, reward, **kwargs) -> None:
        pass
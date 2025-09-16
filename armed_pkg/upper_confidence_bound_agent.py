import numpy as np
import sys
sys.path.append('.')
from .__agent_base__ import __AgentBase__
from .upper_confidence_bound_action_selection import upperConfidenceBoundActionSelection
class UpperConfidenceBoundAgent(__AgentBase__):
    def __init__(self, k=10, initial=0, c=2):
        super().__init__(k=k, initial=initial)
        self.c = c

    def select_action(self, **kwargs) -> int:
        return upperConfidenceBoundActionSelection(self.q_estimation, self.action_count, np.sum(self.action_count), c=kwargs.get('c', self.c))

    def update_estimation(self, action, reward, **kwargs) -> None:
        self.action_count[action] += 1
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
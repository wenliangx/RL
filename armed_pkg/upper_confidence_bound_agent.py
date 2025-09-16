import numpy as np
import sys
sys.path.append('.')
from .__agent_base__ import __AgentBase__
class UpperConfidenceBoundAgent(__AgentBase__):
    def __init__(self, k=10, initial=0, c=2):
        super().__init__(k=k, initial=initial)
        self.c = c

    def select_action(self, **kwargs) -> np.intp:
        c = kwargs.get('c', self.c)
        tmp = self.q_estimation + c * np.sqrt(np.log(np.sum(self.action_count) + 1) / (self.action_count + 1e-5))
        return np.argmax(tmp)

    def update_estimation(self, action, reward, **kwargs) -> None:
        self.action_count[action] += 1
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
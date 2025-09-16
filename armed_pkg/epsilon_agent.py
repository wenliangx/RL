import numpy as np
import sys
sys.path.append('.')
from .__agent_base__ import __AgentBase__

class EpsilonAgent(__AgentBase__):
    def __init__(self, k=10, initial=0, epsilon=0.1):
        super().__init__(k=k, initial=initial)
        self.epsilon = epsilon

    def select_action(self, **kwargs) -> np.intp:
        epsilon = kwargs.get('epsilon', self.epsilon)
        if np.random.rand() < epsilon:
            return np.intp(np.random.randint(self.k))
        else:
            return np.argmax(self.q_estimation)

    def update_estimation(self, action, reward, **kwargs) -> None:
        self.action_count[action] += 1
        alpha = kwargs.get('alpha', None)
        if alpha is not None:
            self.q_estimation[action] += alpha * (reward - self.q_estimation[action])
        else:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
import numpy as np
import sys
sys.path.append('.')
from .__agent_base__ import __AgentBase__
from .gradient_action_selection import GradientActionSelection

class GradientAgent(__AgentBase__):
    def __init__(self, k=10, initial=0, alpha=0.1, baseline=True):
        super().__init__(k=k, initial=initial)
        self.alpha = alpha
        self.baseline = baseline
        self.gradient_action_selection = GradientActionSelection(k=k, alpha=alpha, baseline=baseline)

    def select_action(self, **kwargs) -> int:
        return self.gradient_action_selection.select_action()

    def update_estimation(self, action, reward, **kwargs) -> None:
        self.gradient_action_selection.updatePreferences(action=action, reward=reward)
        super().update_estimation(action, reward, **kwargs)
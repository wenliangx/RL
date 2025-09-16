from .utils.soft_max_distribution import SoftMaxDistribution
import numpy as np
class GradientActionSelection:
    def __init__(self, k=10, alpha=0.1, baseline=True):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.average_reward = 0.0
        self.action_sum = 0
        self.preferences = np.zeros(k)
        self.distribution = SoftMaxDistribution(k, initial=self.preferences)
    
    def select_action(self) -> int:
        # if last_action_index is None:
        #     return int(np.random.randint(self.k))
        self.action_sum += 1
        # if self.baseline:
        #     self.average_reward += (reward - self.average_reward) / self.action_sum
        # else:
        #     self.average_reward = 0
        # for preference_index, preference in enumerate(self.preferences):
        #     if preference_index != last_action_index:
        #         self.preferences[preference_index] += self.alpha * (reward - self.average_reward) * (1 - self.distribution.distribution[preference_index])
        #     else:
        #         self.preferences[preference_index] -= self.alpha * (reward - self.average_reward) * self.distribution.distribution[preference_index]
        # self.distribution.datas = self.preferences
        return int(np.random.choice(np.arange(self.k), p=self.distribution.distribution))
    
    def updatePreferences(self, action, reward):
        if self.baseline:
            self.average_reward += (reward - self.average_reward) / self.action_sum
        else:
            self.average_reward = 0
        for preference_index, preference in enumerate(self.preferences):
            if preference_index == action:
                self.preferences[preference_index] += self.alpha * (reward - self.average_reward) * (1 - self.distribution.distribution[preference_index])
            else:
                self.preferences[preference_index] -= self.alpha * (reward - self.average_reward) * self.distribution.distribution[preference_index]
        self.distribution.datas = self.preferences
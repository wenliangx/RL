from scipy.special import softmax
import torch.nn.functional as F
import numpy as np
class GradientActionSelection:
    def __init__(self, k=10, alpha=0.1, baseline=True):
        self.k = k
        self.alpha = alpha
        self.baseline = baseline
        self.average_reward = 0.0
        self.action_sum = 0
        self.preferences = np.zeros(k)
        self.distribution = softmax(self.preferences)
    
    def select_action(self) -> int:
        self.action_sum += 1
        return int(np.random.choice(np.arange(self.k), p=self.distribution))
    
    def updatePreferences(self, action, reward):
        if self.baseline:
            self.average_reward += (reward - self.average_reward) / self.action_sum
        else:
            self.average_reward = 0
        for preference_index, preference in enumerate(self.preferences):
            if preference_index == action:
                self.preferences[preference_index] += self.alpha * (reward - self.average_reward) * (1 - self.distribution[preference_index])
            else:
                self.preferences[preference_index] -= self.alpha * (reward - self.average_reward) * self.distribution[preference_index]
        self.distribution = softmax(self.preferences)


import torch
class GradientActionSelectionTorch:
    def __init__(self, k=10, num_env=10, alpha=0.1, baseline=True, device='cpu'):
        self.k = k
        self.num_env = num_env
        self.alpha = alpha
        self.baseline = baseline
        self.average_reward = torch.zeros(num_env, device=device)
        self.action_sum = torch.zeros(num_env, device=device)
        self.preferences = torch.zeros((num_env, k), device=device)
        self.distribution = F.softmax(self.preferences, dim=1)
        self.device = device
    
    def select_action(self) -> torch.Tensor:
        self.action_sum += 1
        return torch.multinomial(self.distribution, num_samples=1).squeeze()
    
    def updatePreferences(self, action, reward):
        if self.baseline:
            self.average_reward += (reward - self.average_reward) / self.action_sum
        else:
            self.average_reward = torch.zeros(self.num_env, device=self.device)
        diff = torch.zeros((self.num_env, self.k), device=self.device)
        diff[torch.arange(self.num_env), action.squeeze()] = 1.0
        self.preferences += self.alpha * (reward - self.average_reward).unsqueeze(1) * (diff - self.distribution)
        self.distribution = F.softmax(self.preferences, dim=1)
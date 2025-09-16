import numpy as np
import torch

class BaseEnvironment:
    def __init__(self, k=10, mean=0, std=1):
        self.k = k
        self.mean = mean
        self.std = std
        self.q_star = np.random.normal(self.mean, self.std, self.k)

    def step(self, action):
        reward = np.random.normal(self.q_star[action], 1)
        return reward

class BaseEnvironmentTorch:
    def __init__(self, k=10, num_env=10, mean=0, std=1, device='cuda'):
        self.k = k
        self.mean = mean
        self.std = std
        self.num_env = num_env
        self.device = device
        self.q_star = torch.normal(
            mean=torch.full((self.num_env, self.k), self.mean, device=self.device, dtype=torch.float),
            std=torch.full((self.num_env, self.k), self.std, device=self.device, dtype=torch.float)
        )

    def step(self, action: torch.Tensor):
        reward = torch.normal(
            mean=self.q_star.gather(1, action.view(-1, 1)).squeeze(),
            std=torch.ones(self.num_env, device=self.q_star.device)
        )
        return reward
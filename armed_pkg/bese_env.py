import numpy as np
class BaseEnvironment:
    def __init__(self, k=10, mean=0, std=1):
        self.k = k
        self.mean = mean
        self.std = std
        self.q_star = np.random.normal(self.mean, self.std, self.k)

    def step(self, action):
        reward = np.random.normal(self.q_star[action], 1)
        return reward


import numpy as np
class EpsilonAgent:
    def __init__(self, k=10, epsilon=0.1, initial=0):
        self.k = k
        self.epsilon = epsilon
        self.initial = initial
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_estimation)

    def update_estimation(self, action, reward):
        self.action_count[action] += 1
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
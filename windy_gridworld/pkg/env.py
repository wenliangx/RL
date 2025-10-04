import numpy as np
from .policy import Policy
class Env:
    def __init__(self):
        self.state = [3,0]
        self._size = (7, 10)
        self._wind = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
        self._terminal = (3, 7)

    def reset(self):
        self.state = [3,0]

    def _limit_x(self):
        self.state[0] = min(max(0, self.state[0]), self._size[0] - 1)

    def _limit_y(self):
        self.state[1] = min(max(0, self.state[1]), self._size[1] - 1)

    def _act(self, action):
        self.state[1] += action[1]
        self._limit_y()
        self.state[0] += action[0]
        self.state[0] -= self._wind[self.state[1]]
        self._limit_x()
        return 0 if self.state[0] == self._terminal[0] and self.state[1] == self._terminal[1] else -1

    def run(self, policy: Policy):
        last_action_index, last_action = policy.get_action(self.state.copy())
        reward =  -1
        times = 0
        while reward == -1:
            last_state = self.state.copy()
            reward = self._act(last_action)
            last_action_index, last_action = policy.train(last_state, last_action_index, self.state.copy(), reward)
            times += 1
            yield times


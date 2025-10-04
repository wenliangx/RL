import numpy as np
import unittest

up = [-1, 0]
down = [1, 0]
left = [0, -1]
right = [0, 1]



class Policy:
    def __init__(self, alpha: float = 0.01, epsilon: float = 0.9):
        self.q = [[[0 for _ in range(4)]for _ in range(10)] for _ in range (7)]
        self.actions = [up, down, left, right]
        self.alpha = alpha
        self.epsilon = epsilon
        
    def get_action(self, state):
        q_value = self.q[state[0]][state[1]]
        best_action_index = q_value.index(max(q_value))
        probs = [(1-self.epsilon) / 3] * 4
        probs[best_action_index] = self.epsilon
        choice = np.random.choice(a=range(4), size=1, p=probs)[0]
        return choice, self.actions[choice]

    def train(self, last_state, last_action_index, state, reward):
        action_index, action = self.get_action(last_state)
        self.q[last_state[0]][last_state[1]][last_action_index] += \
            self.alpha * (
                reward +
                self.q[state[0]][state[1]][action_index] -
                self.q[last_state[0]][last_state[1]][last_action_index]
            )
        return action_index, action



class TestPolicy(unittest.TestCase):
    def test_q(self):
        policy = Policy()
        self.assertEqual(policy.q[0][0][0], 0)
        self.assertEqual(len(policy.q), 7)
        self.assertEqual(len(policy.q[0]), 10)
        self.assertEqual(len(policy.q[0][0]), 4)
        self.assertEqual(policy.q[0][0], [0,0,0,0])
if __name__ == '__main__':
        unittest.main()
from .env import Env
from .cell import Cell
import numpy as np
from .actions import actions
from .policy import Policy

def policy_evaluation(env: Env, policy: Policy, gamma=0.9, theta=1e-6):
    state_num = env.grid_size[0] * env.grid_size[1]
    V = np.zeros(state_num)

    while True:
        delta = 0
        for state in range(state_num):
            v = 0
            for action in actions:
                action_prob = policy.getProbability(env.index_to_state(state),action=action)
                p = env.get_p(state, action)
                prob = p[0]['prob']
                next_state = env.state_to_index(p[0]['next_state'])
                reward = p[0]['reward']
                v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[state]))
            V[state] = v
        if delta < theta:
            break
    return V
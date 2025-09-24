from .env import Env
from .cell import Cell
import numpy as np
from .actions import actions
from .policy import Policy

def policy_improvement(env: Env, policy: Policy, v_s: np.ndarray, gamma=0.9):
    policy_stable = True
    for s in range(env.state_num):
        old_action = policy.probabilities[env.index_to_state(s).x, env.index_to_state(s).y].copy()
        action_values = np.zeros(len(actions))
        for action in actions:
            p = env.get_p(s, action)
            for i in p:
                prob, next_state, reward = i['prob'], i['next_state'], i['reward']
                action_values[action.index] += prob * (reward + gamma * v_s[env.state_to_index(next_state)])
        best_action = np.argwhere(action_values == np.max(action_values)).flatten()
        new_action = np.zeros(len(actions))
        new_action[best_action] = 1 / len(best_action)
        policy.probabilities[env.index_to_state(s).x, env.index_to_state(s).y] = new_action
        if not np.array_equal(old_action, new_action):
            policy_stable = False
    return policy_stable, policy
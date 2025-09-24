from .env import Env
from .cell import Cell
import numpy as np
from .actions import actions
from .policy import Policy


def estimatePolicyValueByLinearEquation(env: Env, policy: Policy, gamma=0.9):
    state_num = env.grid_size[0] * env.grid_size[1]
    value = np.zeros(state_num)
    A = np.zeros((state_num, state_num))
    b = np.zeros((state_num, 1))


    for x in range(env.grid_size[0]):
        for y in range(env.grid_size[1]):
            A[env.state_to_index(Cell(x, y)), env.state_to_index(Cell(x, y))] = 1.0
            for action in actions:
                transitions = env.get_p(Cell(x, y), action)
                for transition in transitions:
                    prob = transition['prob']
                    next_state = transition['next_state']
                    reward = transition['reward']
                    A[env.state_to_index(Cell(x, y)), env.state_to_index(next_state)] -= policy.getProbability(Cell(x, y),action) * gamma * prob
                    b[env.state_to_index(Cell(x, y)), 0] += policy.getProbability(Cell(x, y),action) * prob * reward

    
    value = np.linalg.solve(A, b)
    return value
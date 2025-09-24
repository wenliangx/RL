from .env import Env, SpecialCellPair
from .cell import Cell
import numpy as np
from .actions import actions

def estimateBsetPolicyValueByLinearEquation(env, gamma=0.9, theta=1e-6):
    state_num = env.grid_size[0] * env.grid_size[1]
    V = np.zeros(state_num)

    while True:
        delta = 0
        for x in range(env.grid_size[0]):
            for y in range(env.grid_size[1]):
                s = Cell(x, y)
                idx = env.state_to_index(s)
                v = V[idx]
                V[idx] = max(
                    sum(
                        t['prob'] * (t['reward'] + gamma * V[env.state_to_index(t['next_state'])])
                        for t in env.get_p(s, a)
                    )
                    for a in actions
                )
                delta = max(delta, abs(v - V[idx]))
        if delta < theta:
            break
    return V

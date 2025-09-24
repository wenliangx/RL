import sys
sys.path.append('/ws/gridworld')
from pkg.env import Env, SpecialCellPair
from pkg.cell import Cell
import numpy as np
from pkg.actions import up, down, left, right, actions
from pkg.policy import Policy


def estimate(env: Env, policy: Policy, gamma=0.9):
    state_num = env.grid_size[0] * env.grid_size[1]
    value = np.zeros(state_num)
    A = np.zeros((state_num, state_num))
    b = np.zeros((state_num, 1))

    def index_to_state(index):
        return Cell(index // env.grid_size[1], index % env.grid_size[1])

    def state_to_index(state):
        return state.x * env.grid_size[1] + state.y


    for x in range(env.grid_size[0]):
        for y in range(env.grid_size[1]):
            A[state_to_index(Cell(x, y)), state_to_index(Cell(x, y))] = 1.0
            for action in actions:
                transitions = env.get_p(Cell(x, y), action)
                for transition in transitions:
                    prob = transition['prob']
                    next_state = transition['next_state']
                    reward = transition['reward']
                    A[state_to_index(Cell(x, y)), state_to_index(next_state)] -= policy.getProbability(Cell(x, y),action) * gamma * prob
                    b[state_to_index(Cell(x, y)), 0] += policy.getProbability(Cell(x, y),action) * prob * reward

    
    value = np.linalg.solve(A, b)
    return value

if __name__ == "__main__":
    grid_size=(5, 5)
    special_cell_pair_list = [
        SpecialCellPair(start=Cell(0, 1), terminal=Cell(4, 1), reward=10),
        SpecialCellPair(start=Cell(0, 3), terminal=Cell(2, 3), reward=5)
    ]
    env = Env(grid_size=grid_size, special_cell_pair_list=special_cell_pair_list)
    prbabilities = 0.25 * np.ones((grid_size[0], grid_size[1], len(actions)))
    policy = Policy(grid_size=grid_size, prbabilities=prbabilities)
    value = estimate(env=env, policy=policy, gamma=0.9)
    print(value.reshape(env.grid_size))
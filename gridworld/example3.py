import sys
sys.path.append('/ws/gridworld')
from pkg.env import Env, SpecialCellPair
from pkg.cell import Cell
import numpy as np
from pkg.actions import actions
from pkg.policy import Policy
from pkg.policy_evaluation import policy_evaluation


if __name__ == "__main__":
    grid_size=(5, 5)
    special_cell_pair_list = [
        SpecialCellPair(start=Cell(0, 1), terminal=Cell(4, 1), reward=10),
        SpecialCellPair(start=Cell(0, 3), terminal=Cell(2, 3), reward=5)
    ]
    env = Env(grid_size=grid_size, special_cell_pair_list=special_cell_pair_list)
    prbabilities = 0.25 * np.ones((grid_size[0], grid_size[1], len(actions)))
    policy = Policy(grid_size=grid_size, prbabilities=prbabilities)
    v = policy_evaluation(env, policy)
    print(np.reshape(v, grid_size))
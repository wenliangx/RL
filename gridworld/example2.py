import sys
sys.path.append('/ws/gridworld')
from pkg.env import Env, SpecialCellPair
from pkg.cell import Cell
import numpy as np
from pkg.actions import actions
from pkg.get_best_policy_v_s import estimateBsetPolicyValueByLinearEquation

if __name__ == "__main__":
    grid_size=(5, 5)
    special_cell_pair_list = [
        SpecialCellPair(start=Cell(0, 1), terminal=Cell(4, 1), reward=10),
        SpecialCellPair(start=Cell(0, 3), terminal=Cell(2, 3), reward=5)
    ]
    env = Env(grid_size=grid_size, special_cell_pair_list=special_cell_pair_list)
    value = estimateBsetPolicyValueByLinearEquation(env=env, gamma=0.9)
    print(value.reshape(env.grid_size))
import sys
sys.path.append('/ws/gridworld')
from pkg.env import Env, SpecialCellPair
from pkg.cell import Cell
import numpy as np
from pkg.actions import actions
from pkg.policy import Policy
from pkg.policy_evaluation import policy_evaluation
from pkg.policy_improvement import policy_improvement

if __name__ == "__main__":
    grid_size = (5, 5)
    special_cell_pair_list = [
        SpecialCellPair(start=Cell(0, 1), terminal=Cell(4, 1), reward=10),
        SpecialCellPair(start=Cell(0, 3), terminal=Cell(2, 3), reward=5)
    ]
    env = Env(grid_size=grid_size, special_cell_pair_list=special_cell_pair_list)

    prbabilities = 0.25 * np.ones((grid_size[0], grid_size[1], len(actions)))
    policy = Policy(grid_size=grid_size, prbabilities=prbabilities)

    is_policy_stable = False
    iteration = 0
    while not is_policy_stable:
        iteration += 1
        print(f"\n策略迭代第 {iteration} 次:")
        print(policy)

        # 策略评估

        V = policy_evaluation(env, policy=policy)
        print("状态值函数 V(s):")
        print(V.reshape(grid_size))

        # 策略改进
        is_policy_stable, policy = policy_improvement(env, policy, V)

    print("\n最终收敛的最优策略:")
    print(policy)
    print("对应的状态值函数 V*(s):")
    V = policy_evaluation(env, policy)
    print(V.reshape(grid_size))
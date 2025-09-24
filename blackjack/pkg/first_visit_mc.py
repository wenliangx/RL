from .env import *
from .state import *
from .actions import *
from .policy import *
import numpy as np


def estimatePolicyValueByFirstVisitMC(policy: Policy, num_episodes: int, gamma: float = 1.0):
    V = [[]for _ in range(200)]
    value = [0.0 for _ in range(200)]

    for episode in range(num_episodes):
        # 生成一个完整的回合
        init_state = State(0,0,False)
        init_state.reset()
        env = Env(init_state)
        player_sum_list, dealer_card_list, usable_ace_list, rewards_list = env.run(policy=policy)
        G = 0 
        for t in reversed(range(len(player_sum_list) - 1)):
            state_index = State(player_sum_list[t], dealer_card_list[t], usable_ace_list[t]).index
            reward = rewards_list[t + 1]
            G = gamma * G + reward
            first_visit = True
            for pre_t in reversed(range(t)):
                pre_state_index = State(player_sum_list[pre_t], dealer_card_list[pre_t], usable_ace_list[pre_t]).index
                if pre_state_index == state_index:
                    first_visit = False
                    break
            if first_visit:
                V[state_index].append(G)
        
        for i in range(200):
            if len(V[i]) > 0:
                value[i] = float(np.mean(V[i]))
            else:
                value[i] = 0.0
    return value
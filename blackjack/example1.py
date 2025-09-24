import sys
sys.path.append("/ws/blackjack")
from pkg.env import Env
from pkg.state import State
from pkg.policy import Policy
from pkg.first_visit_mc import estimatePolicyValueByFirstVisitMC
import numpy as np
import matplotlib.pyplot as plt
from pkg.darw import plot_blackjack

class SimplePolicy(Policy):
    def action(self, state: State) -> int:
        if state.player_sum >= 20:
            return 0  # stick
        else:
            return 1  # hit


if __name__ == "__main__":
    init_state = State(player_sum=15, dealer_card=10, usable_ace=False)
    env = Env(init_state=init_state)
    simple_policy = SimplePolicy()
    V = estimatePolicyValueByFirstVisitMC(policy=simple_policy, num_episodes=10000)
    V1 = np.array(V[0:100]).reshape(10, 10)
    V2 = np.array(V[100:200]).reshape(10, 10)
    V_more = estimatePolicyValueByFirstVisitMC(policy=simple_policy, num_episodes=500000)
    V3 = np.array(V_more[0:100]).reshape(10, 10)
    V4 = np.array(V_more[100:200]).reshape(10, 10)
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    plot_blackjack(ax=ax, V=V1)
    plot_blackjack(ax=ax2, V=V2)
    plot_blackjack(ax=ax3, V=V3)
    plot_blackjack(ax=ax4, V=V4)
    plt.savefig("value_heatmap.png")
    plt.close()
    

    

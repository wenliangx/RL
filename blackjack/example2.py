import sys
sys.path.append("/ws/blackjack")
from pkg.env import Env, EnvTorch, SimplePolicy
from pkg.state import State
from pkg.policy import Policy
from pkg.first_visit_mc import estimatePolicyValueByFirstVisitMC
import numpy as np
import matplotlib.pyplot as plt
from pkg.darw import plot_blackjack
import torch

if __name__ == "__main__":
    simple_policy = SimplePolicy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    env = EnvTorch(10000, simple_policy, device=device, gamma=1.0)
    env2 = EnvTorch(500000, simple_policy, device=device, gamma=1.0)
    V = env.run()
    V_more = env2.run()
    V1 = np.array(V[0:100]).reshape(10, 10)
    V2 = np.array(V[100:200]).reshape(10, 10)
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
    plt.savefig("value_heatmap_torch.png")
    plt.close()
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
    env = EnvTorch(500000, simple_policy, device=device,gamma=1.0)
    print(env.run())
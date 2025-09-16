
def epsilonGreedyActionSelection(q_estimation, epsilon=0.1) -> int:
    import numpy as np
    k = len(q_estimation)
    if np.random.rand() < epsilon:
        return int(np.random.randint(k))
    else:
        return int(np.argmax(q_estimation))

import torch
def epsilonGreedyActionSelectionTorch(q_estimation, epsilon=0.1) -> torch.Tensor:
    num_env, k = q_estimation.shape
    random_values = torch.rand(num_env, device=q_estimation.device)
    random_actions = torch.randint(0, k, (num_env,), device=q_estimation.device)
    greedy_actions = torch.argmax(q_estimation, dim=1)
    actions = torch.where(random_values < epsilon, random_actions, greedy_actions)
    return actions
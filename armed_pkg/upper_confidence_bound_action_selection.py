def upperConfidenceBoundActionSelection(q_estimation, action_count, total_count, c=2) -> int:
    import numpy as np
    tmp = q_estimation + c * np.sqrt(np.log(total_count + 1) / (action_count + 1e-5))
    return int(np.argmax(tmp))

import torch
def upperConfidenceBoundActionSelectionTorch(q_estimation, action_count, total_count, c=2) -> torch.Tensor:
    if action_count.shape != q_estimation.shape:
        raise ValueError("action_count and q_estimation must have the same shape")
    total_count = total_count + 1  # To avoid log(0)
    exploration_term = c * torch.sqrt(torch.log(total_count) / (action_count + 1e-5))
    ucb_values = q_estimation + exploration_term
    actions = torch.argmax(ucb_values, dim=1)
    return actions
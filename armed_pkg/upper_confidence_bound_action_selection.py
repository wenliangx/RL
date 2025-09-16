def upperConfidenceBoundActionSelection(q_estimation, action_count, total_count, c=2) -> int:
    import numpy as np
    tmp = q_estimation + c * np.sqrt(np.log(total_count + 1) / (action_count + 1e-5))
    return int(np.argmax(tmp))

def epsilonGreedyActionSelection(q_estimation, epsilon=0.1) -> int:
    import numpy as np
    k = len(q_estimation)
    if np.random.rand() < epsilon:
        return int(np.random.randint(k))
    else:
        return int(np.argmax(q_estimation))
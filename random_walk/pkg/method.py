import numpy as np
from typing import Generator, Tuple, Union
from itertools import accumulate, chain


def batch(episodes: Generator[list[Tuple[int, int, float, float]], None, None]):
    result = []
    for episode in episodes:
        result += episode
        # for item in episode:
        #     result += item
        yield result
    # yield [item for episode in episodes for item in episode]

def cal_RMS(estimate):
    true_value = (np.array(list(range(5))) + 1.0) / 6.0
    estimate = np.array(estimate[1:6])
    return np.sqrt(np.mean([(estimate[i] - true_value[i])**2 for i in range(len(estimate))]))

def monte_carlo_estimate(
        episodes:
            Generator[list[Tuple[int, int, float, float]], None, None],
        alpha=0.01
):
    length = 7
    estimate = [0.5] * length
    estimate[0] = 0.0
    estimate[-1] = 0.0
    yield {"estimate": estimate, "RMS": cal_RMS(estimate)}

    for episode in episodes:
        count = [-1.0] * length
        for state, _, _, r in episode:
            count[state] += 1
            if count[state] >= 0:
                estimate[state] += alpha * (r - estimate[state])
        yield {"estimate": estimate, "RMS": cal_RMS(estimate)}

def dt_estimate(
        episodes: 
            Generator[list[Tuple[int, int, float, float]], None, None],
        alpha=0.1
):
    length = 7
    estimate = [0.5] * length
    estimate[0] = 0.0
    estimate[-1] = 0.0
    yield {"estimate": estimate, "RMS": cal_RMS(estimate)}

    for episode in episodes:
        for state, next_state, reward, r in episode:
            estimate[state] += alpha * (reward + estimate[next_state] - estimate[state])
        yield {"estimate": estimate, "RMS": cal_RMS(estimate)}


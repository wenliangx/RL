import numpy as np
from typing import Generator, Tuple
from itertools import accumulate, chain

def _simulate(start_id: int=3)  -> Tuple[list[int], list[float]]:
    assert 0 <= start_id <= 6, "id must be between 0 and 6"
    id = start_id
    states, rewards = [id], [0.0]
    while id not in (0, 6):
        id += int(np.random.choice([-1, 1]))
        states.append(id)
        rewards.append(1.0 if id == 6 else 0.0)
    return states, rewards

def simulate(start_id: int = 3, sim_times: int = 1) -> (
    Generator[list[Tuple[int, int, float, float]], None, None]
):
    for _ in range(sim_times):
        result = _simulate(start_id)
        returns = list(reversed(list(accumulate(reversed(result[1])))))
        yield [(result[0][i], result[0][i+1], result[1][i+1], returns[i+1]) for i in range(len(result[0]) - 1)]


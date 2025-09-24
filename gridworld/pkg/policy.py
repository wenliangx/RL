from .cell import Cell
from .actions import up, down, left, right, actions, Action
import numpy as np


class Policy:
    def __init__(self, grid_size, prbabilities: np.ndarray):
        self.grid_size = grid_size
        if prbabilities.shape != (grid_size[0], grid_size[1], 4):
            raise ValueError("Probabilities array must have shape (grid_size[0], grid_size[1], 4)")
        self.probabilities = prbabilities
        

    def getProbability(self, state: Cell, action: Action):
        return self.probabilities[state.x][state.y][action.index]

    
    def __str__(self):
        action_symbols = ['↑', '↓', '←', '→']
        result = ""
        for i in range(self.grid_size[0]):
            row = []
            for j in range(self.grid_size[1]):
                probs = self.probabilities[i, j]
                if np.allclose(probs, 0):
                    row.append(" . ")
                else:
                    best = np.argwhere(probs == np.max(probs)).flatten()
                    # 如果有多个最大概率动作，全部显示
                    symbol = "".join([action_symbols[idx] for idx in best])
                    row.append(f"{symbol:^3}")
            result += " ".join(row) + "\n"
        return result

    
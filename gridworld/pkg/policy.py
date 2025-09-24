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
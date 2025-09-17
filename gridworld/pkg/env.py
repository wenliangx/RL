from .cell import Cell

class SpecialCellPair:
    def __init__(self, start: Cell, terminal: Cell, reward):
        self.start = start
        self.terminal = terminal
        self.reward = reward


class Env:
    def __init__(self, grid_size=(5, 5), special_cell_pair_list=None):
        self.grid_size = grid_size
        self.special_cell_pair_list = special_cell_pair_list if special_cell_pair_list is not None else []
        self.special_reward = [pair.reward for pair in self.special_cell_pair_list]

    @property
    def reward_set(self):
        return set(self.special_reward + [0, -1])

    def get_p(self, state: Cell, action):
        for pair in self.special_cell_pair_list:
            if state == pair.start:
                return [{'prob': 1.0, 'next_state': pair.terminal, 'reward': pair.reward}]
        
        next_state = state + action
        if 0 <= next_state.x < self.grid_size[0] and 0 <= next_state.y < self.grid_size[1]:
            return [{'prob': 1.0, 'next_state': next_state, 'reward': 0}]
        else:
            return [{'prob': 1.0, 'next_state': state, 'reward': -1}]

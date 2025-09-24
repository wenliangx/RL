from .cell import Cell

class Action:
    def __init__(self, cell , index):
        self.cell = cell
        self.index = index
    def __Int__(self):
        return self.index


up = Action(Cell(-1, 0), 0)
down = Action(Cell(1, 0), 1)
left = Action(Cell(0, -1), 2)
right = Action(Cell(0, 1), 3)

actions = [up, down, left, right]

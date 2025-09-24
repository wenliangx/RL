import numpy as np
class State:
    def __init__(self, player_sum: int, dealer_card: int, usable_ace: bool):
        self.player_sum = player_sum
        self.dealer_card = dealer_card
        self.usable_ace = usable_ace
    
    def copy(self):
        return State(self.player_sum, self.dealer_card, self.usable_ace)

    def reset(self):
        self.player_sum = np.random.randint(12,22,size=1)[0]
        self.dealer_card = np.random.randint(1,11,size=1)[0]
        self.usable_ace = np.random.choice([True, False], p=[0.5, 0.5])

    @property
    def index(self):
        return (self.player_sum - 12) + (self.dealer_card - 1) * 10 + int(self.usable_ace) * 100 


def addPlayerCard(state: State, card: int) -> State:
    if(card < 1 or card > 13):
        raise ValueError("Card must be between 1 and 13")
    if(card > 10):
        card = 10
    new_state = state.copy()
    new_state.player_sum += card
    if new_state.player_sum > 21 and new_state.usable_ace:
        new_state.player_sum -= 10
        new_state.usable_ace = False
    return new_state


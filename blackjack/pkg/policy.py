from abc import ABC, abstractmethod
from .state import State
from .actions import stick, hit, actions
import torch

class Policy(ABC):
    @abstractmethod
    def action(self, state: State) -> int:
        pass

class PolicyTorch(ABC):
    @abstractmethod
    def action(self, player_sum: torch.Tensor, dealer_card: torch.Tensor, usable_ace: torch.Tensor) -> torch.Tensor:
        pass
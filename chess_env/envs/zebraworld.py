from typing import Optional
import numpy as np
import gymnasium as gym
from . import ChessWorldEnv

class ZebraWorldEnv(ChessWorldEnv):

    def __init__(self, width: int = 5, height: int = 5, target_loc = (1,3)):
        super().__init__(width, height, target_loc)

        # Define what actions are available (8 directions + no-op)
        self.action_space = gym.spaces.Discrete(9)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([-3, 2]),   # Move 3 up, 2 right (row - 3, column + 2)
            1: np.array([-3, -2]),  # Move 3 up, 2 left (row - 3, column - 2)
            2: np.array([-2, 3]),  # Move 3 right, 2 up (column + 3, row - 2)
            3: np.array([2, 3]),   # Move 3 right, 2 down (column + 3, row + 2)
            4: np.array([-2, -3]),  # Move 3 left, 2 up (column - 3, row - 2)
            5: np.array([2, -3]), # Move 3 left, 2 down (column - 3, row + 2)
            6: np.array([3, 2]),   # Move 3 down, 2 right (row + 3, column + 2)
            7: np.array([3, -2]),  # Move 3 down, 2 left (row + 3, column - 2)
            8: np.array([0, 0]),   # don't move
        }

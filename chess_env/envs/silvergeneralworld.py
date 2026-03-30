from typing import Optional
import numpy as np
import gymnasium as gym
from . import ChessWorldEnv

class SilverGeneralWorldEnv(ChessWorldEnv):

    def __init__(self, width: int = 5, height: int = 5, target_loc = (1,3)):
        super().__init__(width, height, target_loc)

        # Define what actions are available (5 directions + no-op)
        self.action_space = gym.spaces.Discrete(6)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Move up (row - 1)
            1: np.array([-1, 1]),  # Move up and right (row - 1, column + 1)
            2: np.array([-1, -1]), # Move up and left (row - 1, column - 1)
            3: np.array([1, 1]),   # Move down and right (row + 1, column + 1)
            4: np.array([1, -1]),  # Move down and left (row + 1, column - 1)
            5: np.array([0, 0]),   # don't move
        }

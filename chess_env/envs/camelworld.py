from typing import Optional
import numpy as np
import gymnasium as gym
from . import ChessWorldEnv

class CamelWorldEnv(ChessWorldEnv):

    def __init__(self, width: int = 5, height: int = 5, target_loc = (1,3)):
        super().__init__(width, height, target_loc)

        # Define what actions are available (8 directions + no-op)
        self.action_space = gym.spaces.Discrete(9)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([-2, 0]) + np.array([-1,1]),   # Move 2 up, 1 right and up (row - 2, column + 1)
            1: np.array([-2, 0]) + np.array([-1,-1]),  # Move 2 up, 1 left and up (row - 2, column - 1)
            2: np.array([0, 2]) + np.array([-1,1]),  # Move 2 right, 1 up and right (column + 2, row - 1)
            3: np.array([0, 2]) + np.array([1,1]),  # Move 2 right, 1 down and right (column + 2, row + 1)
            4: np.array([0, -2]) + np.array([-1,-1]),  # Move 2 left, 1 up and left (column - 2, row - 1)
            5: np.array([0, -2]) + np.array([1,-1]), # Move 2 left, 1 down and left (column - 2, row + 1)
            6: np.array([2, 0]) + np.array([1,1]),   # Move 2 down, 1 right and down (row + 1, column + 1)
            7: np.array([2, 0]) + np.array([1,-1]),  # Move 2 down, 1 left and down (row + 1, column - 1)
            8: np.array([0, 0]),   # don't move
        }

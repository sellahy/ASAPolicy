from typing import Optional
import numpy as np
import gymnasium as gym
from . import ChessWorldEnv

class DragonKingWorldEnv(ChessWorldEnv):

    def __init__(self, width: int = 5, height: int = 5, target_loc = (1,3)):
        super().__init__(width, height, target_loc)

        # Define what actions are available (8 directions + no-op)
        self.action_space = gym.spaces.Discrete(9)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]) * (self.width - 1 - self._agent_location[1]),   # Move as many tiles right as possible without overflowing. If at the edge, don't move
            1: np.array([-1, 0]) * (-self._agent_location[0]),  # Move as many tiles up as possible without overflowing. If at the edge, don't move
            2: np.array([0, -1]) * (-self._agent_location[1]),  # Move as many tiles left as possible without overflowing. If at the edge, don't move
            3: np.array([1,0]) * (self.height - 1 - self._agent_location[0]),   # Move as many tiles down as possible without overflowing. If at the edge, don't move
            4: np.array([-1, 1]),  # Move up and right (row - 1, column + 1)
            5: np.array([-1, -1]), # Move up and left (row - 1, column - 1)
            6: np.array([1, 1]),   # Move down and right (row + 1, column + 1)
            7: np.array([1, -1]),  # Move down and left (row + 1, column - 1)
            8: np.array([0, 0]),   # don't move
        }

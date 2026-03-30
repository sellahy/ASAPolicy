from typing import Optional
import numpy as np
import gymnasium as gym
from . import ChessWorldEnv

class DragonHorseWorldEnv(ChessWorldEnv):

    def __init__(self, width: int = 5, height: int = 5, target_loc = (1,3)):
        super().__init__(width, height, target_loc)

        # Define what actions are available (8 directions + no-op)
        self.action_space = gym.spaces.Discrete(9)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
            4: np.array([-1, 1])  * min(-self._agent_location[0],                  self.width - 1 - self._agent_location[1]), # Move as many tiles as possible diagonally up and right. If at edge, don't move #FIXME: only moves 1 tile
            5: np.array([-1, -1]) * min(-self._agent_location[0],                  -self._agent_location[1]),                 # Move as many tiles as possible diagonally up and left. If at edge, don't move #FIXME: only moves 1 tile
            6: np.array([1, 1])   * min(self.height - 1 - self._agent_location[0], self.width - 1 - self._agent_location[1]), # Move as many tiles as possible diagonally down and right. If at edge, don't move #FIXME doesn't do anything
            7: np.array([1, -1])  * min(self.height - 1 - self._agent_location[0], -self._agent_location[1]),                 # Move as many tiles as possible diagonally down and left. If at edge, don't move #FIXME: only moves 1 tile
            8: np.array([0, 0]),   # don't move
        }

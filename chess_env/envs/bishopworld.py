from typing import Optional
import numpy as np
import gymnasium as gym
from . import ChessWorldEnv

class BishopWorldEnv(ChessWorldEnv):

    def __init__(self, width: int = 5, height: int = 5, target_loc = (1,3)):
        super().__init__(width, height, target_loc)

        # Define what actions are available (4 directions + no-op)
        self.action_space = gym.spaces.Discrete(5)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([-1, 1]),  # Move up and right (row - 1, column + 1)
            1: np.array([-1, -1]), # Move up and left (row - 1, column - 1)
            2: np.array([1, 1]),   # Move down and right (row + 1, column + 1)
            3: np.array([1, -1]),  # Move down and left (row + 1, column - 1)
            4: np.array([0, 0]),   # don't move
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self.timestep = 0

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, np.array([self.height, self.width]), size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        # also ensure agent can reach target (i.e. is bishop is on "white" tile, 
        # target must be on "white" tile. Target and agent pos must be equivalent mod 2)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location) or \
              (sum(self._target_location) % 2 != sum(self._agent_location) % 2):
            self._target_location = self.np_random.integers(
                0, np.array([self.height, self.width]), size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

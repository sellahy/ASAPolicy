from typing import Optional
import numpy as np
import gymnasium as gym
from . import ChessWorldEnv

class BishopWorldEnv(ChessWorldEnv):

    def __init__(self, width: int = 5, height: int = 5, target_loc = (1,3)):
        super().__init__(width, height, target_loc)

        # A bishop can only reach cells of the same (row+col)%2 parity as its
        # starting square. Validate upfront that at least one reachable starting
        # cell exists for the given target_loc.
        assert len(self._same_parity_cells(self._target_location)) > 0, (
            f"target_loc {target_loc} leaves no reachable starting positions "
            f"on a {height}x{width} board."
        )

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

    def _same_parity_cells(self, reference_loc: np.ndarray) -> np.ndarray:
        """Return all grid cells that share the same (row+col)%2 parity as
        reference_loc, excluding reference_loc itself.

        Used in __init__() to validate the target is reachable, and in reset()
        to constrain agent placement to cells the bishop can actually reach.
        """
        parity: int = int(sum(reference_loc) % 2)
        return np.array([
            [r, c] for r in range(self.height) for c in range(self.width)
            if (r + c) % 2 == parity
            and not np.array_equal([r, c], reference_loc)
        ])

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

        # Randomly place the agent on a cell of the same parity as the target.
        # also ensure agent can reach target (i.e. if bishop is on "white" tile,
        # target must be on "white" tile. Target and agent pos must be equivalent mod 2)
        same_parity_cells: np.ndarray = self._same_parity_cells(self._target_location)
        idx: int = self.np_random.integers(0, len(same_parity_cells))
        self._agent_location = same_parity_cells[idx]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

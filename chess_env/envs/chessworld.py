from typing import Optional
import numpy as np
import gymnasium as gym


class ChessWorldEnv(gym.Env):

    def __init__(self, width: int = 5, height: int = 5, target_loc = (1,3)):
        # The size of the square grid (10x10 by default)
        self.width = width # number of columns
        self.height = height # number of rows

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array(target_loc, dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        #self.observation_space = gym.spaces.Dict(
        #    {
        #        "agent": gym.spaces.Box(0, np.array([height, width])-1, shape=(2,), dtype=np.int32),   # [x, y] coordinates
        #        "target": gym.spaces.Box(0, np.array([height, width])-1, shape=(2,), dtype=np.int32),  # [x, y] coordinates
        #        "matrix": gym.spaces.Box(0, 2, shape=(self.height, self.width), dtype=np.int32)
        #    }
        #)
        self.observation_space = gym.spaces.Box(0, 2, shape=(self.height, self.width), dtype=np.int32)

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(5)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
            4: np.array([0, 0]),   # Don't move
        }
        
        self.timestep = 0
    
    truncation_point = 500

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions and 2d observation matrix. 
                  Observation matrix is a matrix of 0's with shape (width, height) and 
                  has a 1 where the agent is and a 2 where the goal is. If agent has
                  reached goal, only the 1 will be on the board.
                  Agent position has key "agent", target location has key "target", 
                  and observation matrix has key "matrix"
        """

        obs_mat = np.zeros([self.height, self.width], dtype='int32')
        obs_mat[self._target_location[0], self._target_location[1]] = 2
        obs_mat[self._agent_location[0], self._agent_location[1]] = 1

        # return {"agent": self._agent_location, "target": self._target_location, "matrix": obs_mat}
        return obs_mat

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
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

        # Randomly place agent, ensuring it's different from target position
        self._agent_location = self._target_location
        while np.array_equal(self._target_location, self._agent_location):
            self._agent_location = self.np_random.integers(
                0, np.array([self.height, self.width]), size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-4 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-4) to a movement direction
        direction = self._action_to_direction[action]

        # Update agent position, over or underflowing to opposite side of 
        # board if agent moves outside grid bounds
        self._agent_location = np.mod(self._agent_location + direction, np.array([self.height, self.width]))

        self.timestep += 1

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # truncate if too many timesteps have been taken
        if self.timestep >= self.truncation_point:
            truncated = True
        else:
            truncated = False

        # reward structure: -1 for each tile away from the goal
        # requires accounting for overflow/underflow to opposite side of board
        # by considering distance to location of goal shifted in every direction
        # to calculate this, for each shifted target location, the distance = max(abs(target_loc - pos_loc)). Take the min of these dists
        shifts = np.array([[1,1],[-1,1],[0,1],[1,0],[-1,0],[0,0],[1,-1],[-1,-1],[0,-1]]) # all permutations of shifts. This is a stupid way of doing this, but I couldn't find a better way to do this in numpy
        shifts = np.multiply(shifts, np.array([self.height, self.width])) # need to shift by height and width to account for under/overflow
        shifted_target_locs = self._target_location + shifts
        abs_coordinatewise_dists = abs(shifted_target_locs - self._agent_location)
        
        # the max of the abs coordinatewise distances is the total number of cardinal moves + diagonal moves
        dists = np.max(abs_coordinatewise_dists, 1)
        closest_dist = np.min(dists)
        reward = -closest_dist.item()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

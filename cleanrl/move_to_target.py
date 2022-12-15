import gym
from gym import spaces
import random
import numpy as np

# Size of one side of the square grid.
GRID_SIZE = 50.0
# Maximum allowed coordinate to remain inside the grid.
MAX = GRID_SIZE * 0.5
# Size of the target to hit.
TARGET_SIZE = 1.0


class MoveEnv(gym.Env):
    """
       Environment where the goal is to move to the target in a grid without
    moving out of the grid.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MoveEnv, self).__init__

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            np.array([-MAX, -MAX, -MAX, -MAX], dtype=np.float32),
            np.array([MAX, MAX, MAX, MAX], dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def step(self, action):
        move = (0.0, 0.0)
        match action:
            case 0:
                pass
            case 1:
                move = (-1.0, 0.0)
            case 2:
                move = (1.0, 0.0)
            case 3:
                move = (0.0, -1.0)
            case 4:
                move = (0.0, 1.0)
        self.x += move[0]
        self.y += move[1]

        out_of_bounds = abs(self.x) > MAX or abs(self.y) > MAX
        dx = self.x - self.target_x
        dy = self.y - self.target_y
        hit = dx**2 + dy**2 < TARGET_SIZE**2
        done = out_of_bounds or hit
        reward = 0.0
        if out_of_bounds:
            reward -= 1.0
        if hit:
            reward += 1.0

        return self.obs(), reward, done, {}

    def reset(self):
        self.x = self.rand_coord()
        self.y = self.rand_coord()
        self.target_x = self.rand_coord()
        self.target_y = self.rand_coord()

        # Uncomment to make the Env work:
        # self.target_x = 0.0
        # self.target_y = 0.0

        return self.obs()

    def render(self, mode='human', close=False):
        print(f"ai:({self.x}, {self.y}), t:({self.target_x}, {self.target_y}))")

    def rand_coord(self):
        return random.random() * GRID_SIZE - MAX

    def obs(self):
        return np.array(
            [self.x, self.y, self.target_x, self.target_y],
            dtype=np.float32,
        )

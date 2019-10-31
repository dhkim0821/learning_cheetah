import numpy as np
from gym import spaces
import cenv


class PointMassEnv(object):

    def __init__(self):
        self.env = cenv.PointMass(np.zeros(2))
        self.action_dim = 2
        self.action_space = spaces.Box(-np.ones(self.action_dim),
                                       np.ones(self.action_dim),
                                       dtype=np.float32)

        self.observation_dim = 4
        self.observation_space = spaces.Box(-np.ones(self.observation_dim),
                                            np.ones(self.observation_dim),
                                            dtype=np.float32)

    def reset(self):
        self.env.reset()
        self.env.goal = (np.random.rand(2) - 0.5) * 10.0
        return self._observation()

    def step(self, action):
        force = action * 20.0
        self.env.apply_force(force)

        obs = self._observation()
        done = self.env.t > 10.0
        reward = 1.0
        reward -= (0.1 * np.linalg.norm(self.env.pos - self.env.goal)) ** 2
        reward = np.clip(reward, -1.0, 1.0)
        return obs, reward, done, {}

    def _observation(self):
        obs = np.concatenate(((self.env.pos - self.env.goal) / 20.0,
                              self.env.vel / 100.0))
        return obs

    def render(self):
        dist = np.linalg.norm(self.env.pos - self.env.goal)
        print('render: ', self.env.pos, self.env.goal, 'dist = %.4f' % dist)

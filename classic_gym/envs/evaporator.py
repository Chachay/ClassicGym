import time
import sys
import random

import numpy as np
import sympy as sy
from scipy.signal import cont2discrete

import gym
from gym import spaces

from classic_gym.model import EnvModel

class EvaporatorModel(EnvModel):
    def gen_rhe_sympy(self):
        q  = sy.symbols('q:{0}'.format(self.NX))
        u  = sy.symbols('u:{0}'.format(self.NU))

        MAT = sy.Matrix([
           -q[0]*(0.150246233766234*q[1] - 0.0383501298701299*u[0] + 8.51922049350649)/20 + 0.025,
           (-0.025974025974026*u[1]*(3.46788*q[1] + 205.2) + \
            (u[1] + 48.8571428571429)*(-0.150246233766234*q[1] + 0.0383501298701299*u[0] + 1.48077950649351))\
            /(4*(u[1] + 48.8571428571429))
        ])
        
        return MAT

class Evaporator(gym.Env):
    def __init__(self, dT=1, obs = 10):
        self.NX = 2
        self.NU = 2
        self.x = np.zeros(self.NX)
        
        self.dT = dT
        self.obs = obs

        self.model = EvaporatorModel(self.NX, self.NU)

        # Controller Weight
        self.action_space = spaces.Box(low=100, high=400, shape=(2,))
        self.observation_space = spaces.Box(
            low=np.array([0.25, 40]),
            high=np.array([1., 80]),
            dtype=np.float32
        )

        self.reset()
    
    def seed(self, seed=None):
        np.random.seed(seed=seed)

    def reset(self):
        self.x = np.zeros(2)
        self.x[0] = np.random.uniform(0.25, 1)
        self.x[1] = np.random.uniform(40, 80)

        self.time = 0

        return self.observe(None)

    def step(self, u):
        u = np.clip(u, 100, 400)
        dT = self.dT / self.obs
        
        self.x = self.model.step_sim(self.x, u, self.dT, self.obs)
        self.time += 1

        obs = self.observe(None)
        info = {}
        terminal = self._is_terminal(obs)
        reward = self._reward(obs, u)

        return obs, reward, terminal, info

    def observe(self, obs):
        return self.x

    def render(self, mode='human', close=False):
        pass

    def _reward(self, obs, act):
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        crashed = not np.array_equiv(obs_clipped, obs)
        J=23.8176373535448*act[0]+0.6*act[1]+1621.96634461555-86.86696632099708*obs[1]
        return -J

    def _is_terminal(self, obs):
        time = self.time > self.spec.max_episode_steps
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        crashed = not np.array_equiv(obs_clipped, obs)
        return time or crashed

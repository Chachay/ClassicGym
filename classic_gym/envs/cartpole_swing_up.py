import time
import sys
import random

import numpy as np
import sympy as sy
from scipy.signal import cont2discrete

import gym
from gym import spaces

from classic_gym.model import EnvModel
from classic_gym.cost import quadraticCostModel 

class CartModel(EnvModel):
    def __init__(self, l = .8, M =1., m=.1, **kwargs):
        self.l = l
        self.M = M
        self.m = m
        super().__init__(**kwargs)

    def gen_rhe_sympy(self):
        g = 9.8
        l = self.l
        M = self.M
        m = self.m

        q  = sy.symbols('q:{0}'.format(4))
        qd = q[2:4]
        u  = sy.symbols('u:{0}'.format(1))
        
        I = sy.Matrix([[1, 0, 0, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, M + m, l*m*sy.cos(q[1])], 
                      [0, 0, l*m*sy.cos(q[1]), l**2*m]])
        f = sy.Matrix([
                       qd[0], 
                       qd[1],
                       l*m*sy.sin(q[1])*qd[1]**2 + u[0],
                      -g*l*m*sy.sin(q[1])])
        return sy.simplify(I.inv()*f)

class CartPoleSwingUp(gym.Env):
    def __init__(self, l = .8, M =1., m=.1, dT=0.05, obs = 5):
        self.NX = 4
        self.NU = 1

        # x = [x, theta, x_dot, theta_dot]
        self.x = np.zeros(self.NX)
        
        self.l = l
        self.M = M
        self.m = m
        
        self.dT = dT
        self.obs = obs
        
        self.model = CartModel(
                        NX=self.NX, NU=self.NU, 
                        l=l, M=M, m=m
                    )
        
        # Controller Weight
        Q = np.diag([1.,100, 1., 10.]) / 100.
        self.cost = quadraticCostModel(
            Q=Q, R=np.zeros((self.NU, self.NU)), 
            q=np.zeros(self.NX), r=np.zeros(self.NU),
            Q_term=Q, q_term=np.zeros(self.NX),
            x_ref = np.array([0., -np.pi, 0., 0.]),
            NX=self.NX, NU=self.NU
        )

        # Define OpenAI gym properties
        self.action_space = spaces.Box(low=-3, high=3, shape=(1,))
        self.observation_space = spaces.Box(
            low=np.array([-10, -2*np.pi, -10, -10]),
            high=np.array([10,  2*np.pi,  10,  10]),
            dtype=np.float32
        )

        self.reset()
    
    def seed(self, seed=None):
        np.random.seed(seed=seed)

    def reset(self):
        # x = [x, theta, x_dot, theta_dot]
        self.x = np.zeros(4)
        self.x[0] = np.random.uniform(-1., 1.)
        self.x[1] = np.random.uniform(-np.pi/6, np.pi/6)

        self.time = 0

        return self.observe(None)

    def step(self, u):
        u = np.clip(u, -3, 3)

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
        import matplotlib.pyplot as plt
        plt.cla()
        # Cart (Dot)
        plt.plot(self.x[0],  0, "ro")
        # Pendulum
        x = [self.x[0], self.x[0]+np.sin(self.x[1])*self.l]
        y = [0, -np.cos(self.x[1])*self.l]
        plt.plot(x, y, "k-")
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        plt.pause(0.1)

    def _reward(self, obs, act):
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        crashed = not np.array_equiv(obs_clipped, obs)
        J = self.cost.L(obs, act).ravel()[0]
        return np.exp(-( J + 100. * crashed))

    def _is_terminal(self, obs):
        time = self.time > self.spec.max_episode_steps
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        crashed = not np.array_equiv(obs_clipped, obs)
        return time or crashed

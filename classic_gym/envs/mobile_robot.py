import time
import sys
import random

import numpy as np
import sympy as sy
from scipy.signal import cont2discrete

import gym
from gym import spaces

class MobileRobot(gym.Env):
    def __init__(self, dT=0.05, obs = 10):
        self.NX = 3
        self.NU = 2

        self.x = np.zeros(self.NX)
        
        self.dT = dT
        self.obs = obs
        
        self.A, self.B = self.gen_lmodel()
        
        q = sy.symbols('q:{0}'.format(self.NX))
        u = sy.symbols('u:{0}'.format(self.NU))
        self.calc_rhe = sy.lambdify([q,u], self.gen_rhe_sympy(), "numpy")

        # Controller Weight
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(2,))
        self.observation_space = spaces.Box(
            low=np.array([-2, -2, -6]),
            high=np.array([2,  2,  6]),
            dtype=np.float32
        )

        self.reset()
    
    def gen_rhe_sympy(self):
        Wheel_Distance = 0.235 #[m]

        q  = sy.symbols('q:{0}'.format(self.NX))
        u  = sy.symbols('u:{0}'.format(self.NU))

        v = (u[0] + u[1])/2
        omega = (u[0] - u[1])/Wheel_Distance

        MAT = sy.Matrix([
                v*sy.sin(q[2]), 
                v*sy.cos(q[2]),
                omega
            ])
        return MAT
    
    def gen_lmodel(self):
        mat = self.gen_rhe_sympy()
        q = sy.symbols('q:{0}'.format(self.NX))
        u = sy.symbols('u:{0}'.format(self.NU))
        
        A = mat.jacobian(q)
        B = mat.jacobian(u)
        
        return (sy.lambdify([q,u], np.squeeze(A), "numpy"),
                sy.lambdify([q,u], np.squeeze(B), "numpy"))
                
    def gen_dmodel(self, x, u, dT):
        f = self.calc_rhe(x, u).ravel()
        A_c = np.array(self.A(x, u))
        B_c = np.array(self.B(x, u))
        
        g_c = np.atleast_2d(f - A_c@x - B_c@u).T
        B = np.hstack((B_c, g_c))

        A_d, B_d, _, _, _ = cont2discrete((A_c, B, 0, 0), dT)
        g_d = B_d[:,self.NU]
        B_d = B_d[:,:self.NU]

        return A_d, B_d, g_d

    def seed(self, seed=None):
        np.random.seed(seed=seed)

    def reset(self):
        self.x = np.zeros(self.NX)

        # machine state
        self.x[0] = np.random.uniform(-2, 2)
        self.x[1] = np.random.uniform(-2, 2)
        self.x[2] = np.random.uniform(-np.pi, np.pi)

        self.time = 0

        return self.observe(None)

    def step(self, u):
        u = np.clip(u, -0.5, 0.5)

        dT = self.dT / self.obs
        
        for _ in range(self.obs):
            A, B, g = self.gen_dmodel(self.x, u, dT)
            self.x = A@self.x + B @ u +g

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
        ax = plt.gca()
        ax.cla()

        l = 0.15 # length of Arrow

        # Robot
        circle = plt.Circle((self.x[0],self.x[1]), 0.1, color='r', fill=True)
        ax.add_artist(circle)
        plt.arrow(self.x[0], self.x[1], 
                  l*np.cos(self.x[2]), l*np.sin(self.x[2]), 
                  head_width=0.05, head_length=0.1, fc='k', ec='k', zorder=10)
        # Target Arrow
        plt.arrow(0, 0, 
                  l*np.cos(0.), l*np.sin(0.), 
                  head_width=0.05, head_length=0.1, fc='b', ec='b')
        plt.axis('equal')
        plt.xlim(-2.1, 2.1)
        plt.ylim(-2.1, 2.1)
        plt.pause(0.1)

    def _reward(self, obs, act):
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        crashed = not np.array_equiv(obs_clipped, obs)
        J_state = np.sum(self.x**2)
        J_action = np.sum(act**2) * 0.1
        return - (J_state + J_action)

    def _is_terminal(self, obs):
        time = self.time > self.spec.max_episode_steps
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        crashed = not np.array_equiv(obs_clipped, obs)
        return time or crashed

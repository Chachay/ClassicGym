import time
import sys
import random

import numpy as np
import sympy as sy
from scipy.signal import cont2discrete

import gym
from gym import spaces

class CartPoleSwingUp(gym.Env):
    def __init__(self, l = .8, M =1., m=.1, dT=0.05, obs = 5):
        
        # x = [x, theta, x_dot, theta_dot]
        self.x = np.zeros(4)
        
        self.l = l
        self.M = M
        self.m = m
        
        self.dT = dT
        self.obs = obs
        
        self.A, self.B = self.gen_lmodel()
        
        q = sy.symbols('q:{0}'.format(4))
        u = sy.symbols('u')
        self.calc_rhe = sy.lambdify([q,u], self.gen_rhe_sympy())

        # Controller Weight
        self.Q = np.diag([1.,100, 1., 10.]) / 100.
        self.x_ref = np.array([0., -np.pi, 0., 0.])

        # Define OpenAI gym properties
        self.action_space = spaces.Box(low=-3, high=3, shape=(1,))
        self.observation_space = spaces.Box(
            low=np.array([-10, -2*np.pi, -10, -10]),
            high=np.array([10,  2*np.pi,  10,  10]),
            dtype=np.float32
        )

        self.reset()
    
    def gen_rhe_sympy(self):
        g = 9.8
        l = self.l
        M = self.M
        m = self.m

        q  = sy.symbols('q:{0}'.format(4))
        qd = q[2:4]
        u  = sy.symbols('u')
        
        I = sy.Matrix([[1, 0, 0, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, M + m, l*m*sy.cos(q[1])], 
                      [0, 0, l*m*sy.cos(q[1]), l**2*m]])
        f = sy.Matrix([
                       qd[0], 
                       qd[1],
                       l*m*sy.sin(q[1])*qd[1]**2 + u,
                      -g*l*m*sy.sin(q[1])])
        return sy.simplify(I.inv()*f)
    
    def gen_lmodel(self):
        mat = self.gen_rhe_sympy()
        q = sy.symbols('q:{0}'.format(4))
        u = sy.symbols('u')
        
        A = mat.jacobian(q)
        #B = mat.jacobian(u)
        B = mat.diff(u)
        
        return (sy.lambdify([q,u], np.squeeze(A), "numpy"),
                sy.lambdify([q,u], np.squeeze(B), "numpy"))
                
    def gen_dmodel(self, x, u, dT):
        f = self.calc_rhe(x, u[0]).ravel()
        A_c = np.array(self.A(x, u[0]))
        B_c = np.array(self.B(x, u[0]))
        
        g_c = f - A_c@x - B_c*u
        B = np.vstack((B_c, g_c)).T

        A_d, B_d, _, _, _ = cont2discrete((A_c, B, 0, 0), dT)
        g_d = B_d[:,1]
        B_d = B_d[:,0]

        return A_d, B_d, g_d

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
        dT = self.dT / self.obs
        
        for _ in range(self.obs):
            A, B, g = self.gen_dmodel(self.x, u, dT)
            self.x = A@self.x + B * u +g

        self.time += 1

        obs = self.observe(None)
        info = {}
        terminal = self._is_terminal(obs)
        reward = self._reward(obs)

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

    def _reward(self, obs):
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        crashed = not np.array_equiv(obs_clipped, obs)
        J = (obs-self.x_ref).T@self.Q@(obs-self.x_ref)
        return np.exp(-( J + 100. * crashed))

    def _is_terminal(self, obs):
        time = self.time > self.spec.max_episode_steps
        obs_clipped = np.clip(obs, self.observation_space.low, self.observation_space.high)
        crashed = not np.array_equiv(obs_clipped, obs)
        return time or crashed

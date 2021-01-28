import argparse

import gym
import classic_gym
import numpy as np

class DDP(object):
    def __init__(self, cost_model, dynamic_model, T, dT, obs_size, act_size):
        self.cost = cost_model 
        self.model = dynamic_model
        self.dt = dT
        self.time_horizon = T

        self.NX = obs_size
        self.NU = act_size

        self.u_guess = np.zeros((T, self.NU))
        self.max_iter = 100

    def act(self, x0):
        # initialize
        opt_count = 0
        u_guess = self.u_guess.copy()
        x_ref = None

        while opt_count < self.max_iter:
            x_guess, cost, derivates = self.calc_forward(x0, x_ref, u_guess)
            k, K = self.calc_backward(derivates)
            x_guess, u_guess = self.update_trajectory(k, K, x_guess, u_guess)

            opt_count += 1

        self.u_guess[:-1] = u_guess[1:]
        self.u_guess[-1] = u_guess[-1]

        self.max_iter = 10
        return self.u_guess[0]
    
    def update_trajectory(self, k, K, x_guess, u_guess):
        x = np.zeros_like(x_guess)
        x[0] = x_guess[0]

        for i in range(self.time_horizon):
            u_guess[i] = u_guess[i] + k[i] + K[i]@(x[i] - x_guess[i])
            x[i+1] = self.model.calc_RK4(x[i], u_guess[i], self.dt)

        return x, u_guess 

    def calc_forward(self, x0, x_ref, u_guess):
        x_guess = np.zeros((self.time_horizon+1, self.NX))
        x_guess[0] = x0

        cost = 0

        derivates = []

        for i in range(self.time_horizon):
            # Calc dynamics, its jacobian and hessian
            x_guess[i+1] = self.model.calc_RK4(x_guess[i], u_guess[i], self.dt)

            Jx = np.asarray(self.model.Jq(x_guess[i], u_guess[i])) 
            Ju = np.asarray(self.model.Ju(x_guess[i], u_guess[i]))
            Hxx = np.asarray(self.model.Hqq(x_guess[i], u_guess[i]))
            Huu = np.asarray(self.model.Huu(x_guess[i], u_guess[i]))
            Hux = np.asarray(self.model.Huq(x_guess[i], u_guess[i]))

            # Calc cost, its jacobian and hessian
            cost += self.cost.L(x_guess[i], u_guess[i])

            Lx = np.squeeze(self.cost.Lq(x_guess[i], u_guess[i])) 
            Lu = np.squeeze(self.cost.Lu(x_guess[i], u_guess[i]))

            Lxx = np.asarray(self.cost.Lqq(x_guess[i], u_guess[i]))
            Luu = np.asarray(self.cost.Luu(x_guess[i], u_guess[i]))
            Lux = np.asarray(self.cost.Luq(x_guess[i], u_guess[i]))

            derivates.append([Jx, Ju, Hxx, Huu, Hux, Lx, Lu, Lxx, Luu, Lux])

        cost += self.cost.V(x_guess[self.time_horizon])
        Vx = np.squeeze(self.cost.Vq(x_guess[self.time_horizon]))
        Vxx = np.asarray(self.cost.Vqq(x_guess[self.time_horizon]))

        derivates.append([None, None, None, None, None, Vx, None, Vxx, None, None])

        return x_guess, cost, derivates

    def calc_backward(self, derivates):
        Vx = derivates[-1][5]
        Vxx = derivates[-1][7]

        k = np.zeros((self.time_horizon, self.NU))
        K = np.zeros((self.time_horizon, self.NU, self.NX))

        for t in range(self.time_horizon-1, -1, -1):
            Jx, Ju, Hxx, Huu, Hux,\
            Lx, Lu, Lxx, Luu, Lux = derivates[t]
            
            Qx = Lx + Jx.T@Vx
            Qu = Lu + Ju.T@Vx
            Qxx = Lxx.T + Jx.T@Vxx@Jx + np.squeeze(Vx.T@Hxx).T
            Qux = Lux.T + Ju.T@Vxx@Jx + np.squeeze(Vx.T@Hux).T
            Quu = Luu.T + Ju.T@Vxx@Ju + np.squeeze(Vx.T@Huu).T

            k[t] = - np.squeeze(np.linalg.pinv(Quu)@Qu)
            K[t] = - np.linalg.pinv(Quu)@Qux

            Vx = Qx + K[t].T@Quu@k[t]
            Vxx = Qxx + K[t].T@Quu@K[t]

        return k, K

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole-swingup-v0",
        help="Cart pole balancing environment on AgX Dynamics",
    )
    args = parser.parse_args()

    env = gym.make(args.env)
    timestep_limit = env.spec.max_episode_steps 
    obs_space = env.observation_space
    action_space = env.action_space

    print("Env Name:", args.env)
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    T = 30 # time horizon
    dT = env.dT
    controller = DDP(env.cost, env.model, 
                     T, dT, 
                     obs_space.high.shape[0], 
                     action_space.high.shape[0])

    o = env.reset()
    for t in range(timestep_limit):
        act = controller.act(o)
        o, _, _, _ = env.step(act)
        print("step {}: observe: {} act:{}".format(t, o, act))
        if t % 2 == 0:
            env.render()
    
    print("Done")

if __name__ == "__main__":
    main()

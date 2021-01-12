"""
    A training script of SAC
"""
import argparse
from distutils.version import LooseVersion
import functools

import gym
import classic_gym
import numpy as np
import torch
from torch import nn
from torch import distributions

import pfrl
from pfrl.agents import SoftActorCritic
from pfrl import experiments
from pfrl.nn.lmbda import Lambda
from pfrl import replay_buffers
from pfrl import utils

# [Gym Wrappers \| alexandervandekleut\.github\.io](https://alexandervandekleut.github.io/gym-wrappers/)
class NormalizeObsSpace(gym.ObservationWrapper):
    """Normalize a Box action space to [-1, 1]^n."""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=-np.ones_like(env.observation_space.low),
            high=np.ones_like(env.observation_space.low),
        )
    def observation(self, obs):
        n_obs = obs.copy()
        # -> [0, orig_high - orig_low]
        n_obs -= self.env.observation_space.low
        # -> [0, 2]
        n_obs /= (self.env.observation_space.high - self.env.observation_space.low) / 2
        # action is in [-1, 1]
        return n_obs - 1


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--env",
        type=str,
        #default="agx-cartpole-v0",
        default="cartpole-swingup-v0",
        help="Cart pole balancing environment on AgX Dynamics",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2 * 10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--async", action="store_true", help="Async training"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to update model for per SAC iteration.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--policy-output-scale",
        type=float,
        default=1.0,
        help="Weight initialization scale of policy output.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        env = NormalizeObsSpace(env)
        env = pfrl.wrappers.NormalizeActionSpace(env)
        if args.monitor:
            from baselines import bench
            env = bench.Monitor(env, args.outdir, allow_early_resets=True)
            #env = pfrl.wrappers.Monitor(env, args.outdir)
        #if not test:
        #    env = pfrl.wrappers.ScaleReward(env, 1e-3)
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.max_episode_steps 
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    del sample_env

    assert isinstance(action_space, gym.spaces.Box)
    obs_size = obs_space.low.size
    action_size = action_space.low.size

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_size * 2),
        Lambda(squashed_diagonal_gaussian_head),
    )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    torch.nn.init.xavier_uniform_(policy[4].weight, gain=args.policy_output_scale)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=3e-4,
    )

    if args.load or args.load_pretrained:
        if args.load_pretrained:
            raise Exception("Pretrained models are currently unsupported.")
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(utils.download_model("SAC", args.env, model_type="final")[0])
    
    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    elif args.num_envs==1:
        def print_evaluation(env, agent, evaluator, step, eval_score):
            print("print eval", eval_score, step, evaluator.env_get_stats())

        try:
            experiments.train_agent_with_evaluation(
                agent=agent,
                env=make_env(0, False),
                eval_env=make_env(0, True),
                outdir=args.outdir,
                steps=args.steps,
                eval_n_steps=None,
                eval_n_episodes=args.eval_n_runs,
                eval_interval=args.eval_interval,
                save_best_so_far_agent=True,
                evaluation_hooks=(print_evaluation,),
            )
        except:
            import glob
            import os
            print("Stopped---")
            print(glob.glob(os.path.join(args.outdir,'*_except')))

    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=True,
        )

if __name__ == "__main__":
    main()

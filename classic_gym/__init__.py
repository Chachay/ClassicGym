from gym.envs.registration import register

register(
    id='cartpole-swingup-v0',
    entry_point='classic_gym.envs:CartPoleSwingUp',
    max_episode_steps=1200
)

register(
    id='evaporator-v0',
    entry_point='classic_gym.envs:Evaporator',
    max_episode_steps=3600
)

register(
    id='mobile-v0',
    entry_point='classic_gym.envs:MobileRobot',
    max_episode_steps=1200
)

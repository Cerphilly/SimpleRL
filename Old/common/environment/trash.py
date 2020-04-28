from gym.envs.registration import register

register(
    id='InvertedPendulumSwing-v2',
    entry_point='environment.gym_env:InvertedPendulumSwingEnv',
    max_episode_steps=1000,
    reward_threshold=910.0,
)

register(
    id='InvertedDoublePendulumSwing-v2',
    entry_point='environment.gym_env:InvertedDoublePendulumSwingEnv',
    max_episode_steps=1000,
    reward_threshold=910.0,
)
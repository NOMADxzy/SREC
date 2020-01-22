from gym.envs.registration import register

register(
    id='rlabr-v0',
    entry_point='rl_abr.envs:ABRSimEnv',
)
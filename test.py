import gym
from rl_abr.envs.abr import ABRSimEnv
from rl_abr.cache.cache import CacheEnv


past_bitrates = 1
#env = ABRSimEnv(trace_type = 'n_train')
env = CacheEnv(normalize = True, cache_size = 512, unseen_recency = 500)
experiences = 1
actions = 10000

total_total_reward = 0
for i in range(experiences):
    env.reset()
    total_reward = 0
    for j in range(actions):
        action = env.action_space.sample()  # direct action for test
        obs, reward, done, info = env.step(action)
        if obs[0] == 0:
            print(obs)
        total_reward += reward
        if done:
            break
    total_total_reward += total_reward
print("average_total_reward train", float(total_total_reward) / experiences)

# env = ABRSimEnv(trace_type = 'n_test')
# total_total_reward = 0
# for i in range(experiences):
#     env.reset()
#     total_reward = 0
#     for j in range(actions):
#         action = env.action_space.sample()  # direct action for test
#         obs, reward, done, info = env.step(action)
#         total_reward += reward
#         if done:
#             break
#     total_total_reward += total_reward
# print("average_total_reward test", float(total_total_reward) / experiences)
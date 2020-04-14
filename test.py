import gym
from rl_abr.envs.abr import ABRSimEnv
from rl_abr.cache.cache import CacheEnv
from rl_abr.load_balance.load_balance import LoadBalanceEnv
import numpy as np

past_bitrates = 1
# env = ABRSimEnv(trace_type = 'n_train')
# env = CacheEnv(normalize = True, cache_size = 512, unseen_recency = 500)
env = LoadBalanceEnv(normalize=True)
experiences = 10
actions = 100000

total_total_length = []
total_total_reward = []
for i in range(experiences):
    env.reset()
    total_reward = 0
    for j in range(actions):
        action = env.action_space.sample()  # direct action for test
        obs, reward, done, info = env.step(action)
        print(obs)
        total_reward += reward
        if done:
            break
    total_total_length.append(j + 1)
    total_total_reward.append(total_reward)
print("average_total_reward", float(sum(total_total_reward)) / experiences)
print("average episode length", float(sum(total_total_length)) / experiences)
print("total actions", sum(total_total_length))
print("std reward", np.std(np.array(total_total_reward)))
print("std episode length", np.std(np.array(total_total_length)))

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
from __future__ import division

import time

import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc

import train
import buffer
import rl_abr.envs as abr_env

# env = gym.make('BipedalWalker-v2')
# env = gym.make('Pendulum-v0')
env = abr_env.ABRSimEnv(trace_type = 'n_train', obs_chunk_len=1, normalize_obs=True)

MAX_EPISODES = 50000
MAX_STEPS = 5000
MAX_BUFFER = 1000000
# MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.n
A_MAX = 1

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    print('EPISODE :- ', _ep)
    for r in range(MAX_STEPS):
        env.render()
        # time.sleep(1)
        state = np.float32(observation)

        actions, action, strength = trainer.get_exploration_action(state)
        # if _ep%5 == 0:
        # 	# validate every 5th episode
        # 	action = trainer.get_exploitation_action(state)
        # else:
        # 	# get action based on observation, use exploration policy here
        # 	action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(action)
        print(str(strength), float(reward), info)

        # # dont update if this is validation
        # if _ep%50 == 0 or _ep>450:
        # 	continue

        if done:
            new_state = None
        else:
            new_state = np.float32(new_observation)
            # push this exp in ram
            ram.add(state, actions, reward, new_state)

        observation = new_observation

        # perform optimization
        trainer.optimize()
        if done:
            break

    # check memory consumption and clear memory
    gc.collect()
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss)

    if _ep % 100 == 0:
        trainer.save_models(_ep)

print('Completed episodes')

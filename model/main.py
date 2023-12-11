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
env = abr_env.ABRSimEnv(trace_type = 'n_train', obs_chunk_len=4, normalize_obs=True)

MAX_EPISODES = 50000
MAX_STEPS = 5000
MAX_BUFFER = 1000000
# MAX_TOTAL_REWARD = 300
ABR_STATE_DIM = 7
PLACE_STATE_DIM = 3
ABR_DIM = env.action_space[0].n
SPACE_DIM = env.action_space[1].n

A_MAX = 1

ram1 = buffer.MemoryBuffer(MAX_BUFFER)
ram2 = buffer.MemoryBuffer(MAX_BUFFER)
abr_trainer = train.Trainer(ABR_STATE_DIM, ABR_DIM, A_MAX, ram1)
place_trainer = train.Trainer(PLACE_STATE_DIM, SPACE_DIM, A_MAX, ram2)

abr_trainer.load_models(msg="abr", episode=500)
place_trainer.load_models(msg="place", episode=600)

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    print('EPISODE :- ', _ep)
    for r in range(MAX_STEPS):
        env.render()
        # time.sleep(1)
        state = np.float32(observation)

        abr_actions, abr_action, abr_strength = abr_trainer.get_exploration_action(state[:ABR_STATE_DIM])
        place_actions, place_action, place_strength = place_trainer.get_exploration_action(np.asarray([state[0],state[-2],state[-1]]))
        full_action = np.asarray([abr_action, place_action])
        if place_action == 0:
            full_action[0] = max(0, abr_action - 2)
        # if _ep%5 == 0:
        # 	# validate every 5th episode
        # 	action = trainer.get_exploitation_action(state)
        # else:
        # 	# get action based on observation, use exploration policy here
        # 	action = trainer.get_exploration_action(state)

        new_observation, reward, done, info = env.step(full_action)
        print(reward, info)

        # # dont update if this is validation
        # if _ep%50 == 0 or _ep>450:
        # 	continue

        if done:
            new_state = None
        else:
            new_state = np.float32(new_observation)
            # push this exp in ram
            ram1.add(state[:ABR_STATE_DIM], abr_actions, reward[0], new_state[:ABR_STATE_DIM])
            ram2.add(np.asarray([state[0],state[-2],state[-1]]), place_actions, reward[1], np.asarray([new_state[0],new_state[-2],new_state[-1]]))

        observation = new_observation

        # perform optimization
        abr_trainer.optimize()
        # place_trainer.optimize()
        if done:
            break

    # check memory consumption and clear memory
    gc.collect()

    if _ep % 100 == 0:
        abr_trainer.save_models("abr",_ep)
        # place_trainer.save_models("place", _ep)

print('Completed episodes')

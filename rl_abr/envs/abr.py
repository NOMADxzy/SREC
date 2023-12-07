import numpy as np
from collections import deque

import gym
from gym import spaces, logger
from gym.utils import seeding # Currently not trying to replace with this right now
from rl_abr.envs.trace_loader import \
    load_traces, load_chunk_sizes, sample_trace, get_chunk_time

ACTION_DIM = 6
class StateNormalizer(object):
    def __init__(self, obs_space):
        self.shift = obs_space.low
        self.range = obs_space.high - obs_space.low

    def normalize(self, obs):
        return (obs - self.shift) / self.range

    # Used for world model reward calculations
    def unnormalize(self, obs):
        return (obs * self.range) + self.shift

class ABRSimEnv(gym.Env):
    """
    Adapt bitrate during a video playback with varying network conditions.
    The objective is to (1) reduce stall (2) increase video quality and
    (3) reduce switching between bitrate levels. Ideally, we would want to
    *simultaneously* optimize the objectives in all dimensions.

    * STATE *
        [The throughput estimation of the past chunk (chunk size / elapsed time),
        download time (i.e., elapsed time since last action), current buffer ahead,
        number of the chunks until the end, the bitrate choice for the past chunk,
        current chunk size of bitrate 1, chunk size of bitrate 2,
        ..., chunk size of bitrate 5]

        Note: we need the selected bitrate for the past chunk because reward has
        a term for bitrate change, a fully observable MDP needs the bitrate for past chunk

    * ACTIONS *
        Which bitrate to choose for the current chunk, represented as an integer in [0, 4]

    * REWARD *
        At current time t, the selected bitrate is b_t, the stall time between
        t to t + 1 is s_t, then the reward r_t is
        b_{t} - 4.3 * s_{t} - |b_t - b_{t-1}|
        Note: there are different definitions of combining multiple objectives in the reward,
        check Section 5.1 of the first reference below.

    * REFERENCE *
        Section 4.2, Section 5.1
        Neural Adaptive Video Streaming with Pensieve
        H Mao, R Netravali, M Alizadeh
        https://dl.acm.org/citation.cfm?id=3098843

        Figure 1b, Section 6.2 and Appendix J
        Variance Reduction for Reinforcement Learning in Input-Driven Environments.
        H Mao, SB Venkatakrishnan, M Schwarzkopf, M Alizadeh.
        https://openreview.net/forum?id=Hyg1G2AqtQ

        A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP
        X Yin, A Jindal, V Sekar, B Sinopoli
        https://dl.acm.org/citation.cfm?id=2787486
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, **kwargs):
        # observation and action space
        # how many past throughput to report
        self.past_chunk_len = 8

        self.trace_type = "n_train" if "trace_type" not in kwargs else kwargs["trace_type"]
        self.obs_chunk_len = 1 if "obs_chunk_len" not in kwargs else kwargs["obs_chunk_len"]
        self.normalize_obs = False if "normalize_obs" not in kwargs else kwargs["normalize_obs"]
        # self.trace_type = "n_train"
        #
        # for key, arg in kwargs.items():
        #     if key == "obs_chunk_len":
        #         self.obs_chunk_len = arg
        #         assert arg < self.past_chunk_len
        #     elif key == "trace_type":
        #         self.trace_type = arg
        self.setup_space()
        # set up seed NOTE: Updated this to pass None into Gym Seeding function, really do need to check this
        self.seed(None)
        # self.seed(42)
        # load all trace files
        self.all_traces = load_traces(self.trace_type)
        # load all video chunk sizes
        self.chunk_sizes = load_chunk_sizes()
        # mapping between action and bitrate level
        self.bitrate_map = [0.3, 0.75, 1.2, 1.85, 2.85, 4.3]  # Mbps
        # assert number of chunks for different bitrates are all the same
        assert len(np.unique([len(chunk_size) for \
               chunk_size in self.chunk_sizes])) == 1
        self.total_num_chunks = len(self.chunk_sizes[0])

    def observe(self):
        if self.chunk_idx < self.total_num_chunks:
            valid_chunk_idx = self.chunk_idx
        else:
            valid_chunk_idx = 0

        if self.past_action is not None:
            valid_past_action = self.past_action
        else:
            valid_past_action = 0

        # network throughput of past chunk, past chunk download time,
        # current buffer, number of chunks left and the last bitrate choice
        # obs_arr = [self.past_chunk_throughputs[-1],
        #            self.past_chunk_download_times[-1],
        #            self.buffer_size,
        #            self.total_num_chunks - self.chunk_idx,
        #            valid_past_action]
        obs_arr = [self.past_chunk_throughputs[i] for i in range(self.past_chunk_len - self.obs_chunk_len,self.past_chunk_len)]
        obs_arr.extend([self.past_chunk_download_times[-1],
                   self.buffer_size,
                   # self.total_num_chunks - self.chunk_idx,
                   valid_past_action])

        # current chunk size of different bitrates
        # TODO: Should change this to match chunk indices
        # obs_arr.extend(self.chunk_sizes[i][valid_chunk_idx] for i in range(6))

        # fit in observation space
        for i in range(len(obs_arr)):
            if obs_arr[i] > self.og_obs_high[i]:
                logger.warn('Observation at index ' + str(i) +
                    ' at chunk index ' + str(self.chunk_idx) +
                    ' has value ' + str(obs_arr[i]) +
                    ', which is larger than obs_high ' +
                    str(self.og_obs_high[i]))
                obs_arr[i] = self.og_obs_high[i]

        obs_arr = np.array(obs_arr)
        # assert self.og_observation_space.contains(obs_arr)

        return obs_arr

    def reset(self):
        self.trace, self.curr_t_idx = \
            sample_trace(self.all_traces, self.np_random)
        self.chunk_time_left = get_chunk_time(
            self.trace, self.curr_t_idx)
        self.chunk_idx = 0
        self.buffer_size = 0.0  # initial download time not counted
        self.past_action = None
        self.past_chunk_throughputs = deque(maxlen=self.past_chunk_len)
        self.past_chunk_download_times = deque(maxlen=self.past_chunk_len)
        for _ in range(self.past_chunk_len):
            self.past_chunk_throughputs.append(0)
            self.past_chunk_download_times.append(0)

        self.past_observation = self.observe()
        return self.past_observation if not self.normalize_obs else self.state_normalizer.normalize(self.past_observation)

    def render(self, mode='human'):
        return

    def seed(self, seed):
        # TODO: Make this match the signature of OpenAI Gym
        # Note: Attempted to use first result for seed, rng from the OpenAI gym utils function, but the int_list_ .. appeared to cause issues
        self.np_random = seeding.np_random(seed)[0]
        # self.np_random = self.np_random_func(seed)

    def np_random_func(self, seed):
        if not (isinstance(seed, int) and seed >= 0):
            raise ValueError('Seed must be a non-negative integer.')
        rng = np.random.RandomState()
        rng.seed(seed)
        # Note: this uses the Park supplied np_random function, not the function from OpenAI gym, supplied here
        #rng.seed(_int_list_from_bigint(hash_seed(seed)))
        return rng

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.og_obs_low = np.array([0] * (4 + self.obs_chunk_len - 1))
        # NOTE: NEED TO FIX
        # added variation for the first past observation values
        self.og_obs_high = np.concatenate((np.array([10e6] * (self.obs_chunk_len)), np.array([10, 40, 5])))

        self.og_observation_space = spaces.Box(
            low=self.og_obs_low, high=self.og_obs_high, dtype=np.float32)
        if self.normalize_obs:
            self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape=(len(self.og_obs_low),), dtype=np.float32)
        else:
            self.observation_space = self.og_observation_space
        self.state_normalizer = StateNormalizer(self.og_observation_space)

        self.action_space = spaces.Discrete(ACTION_DIM)

    def step(self, action):
        # Note: if this function runs when the environment is done, it will cause an error, this should not occur

        # 0 <= action < num_servers
        assert self.action_space.contains(action)

        # Note: sizes are in bytes, times are in seconds
        chunk_size = self.chunk_sizes[action][self.chunk_idx]

        # compute chunk download time based on trace
        delay = 0  # in seconds

        # keep experiencing the network trace
        # until the chunk is downloaded
        while chunk_size > 1e-8:  # floating number business

            throuput = self.trace[1][self.curr_t_idx] / 8.0 * 1e6  # bytes/second

            chunk_time_used = min(self.chunk_time_left, chunk_size / throuput)

            chunk_size -= throuput * chunk_time_used
            self.chunk_time_left -= chunk_time_used
            delay += chunk_time_used

            if self.chunk_time_left == 0: # 当前加载失败

                self.curr_t_idx += 1
                if self.curr_t_idx == len(self.trace[1]):
                    self.curr_t_idx = 0

                self.chunk_time_left = get_chunk_time(self.trace, self.curr_t_idx)

        # compute buffer size
        rebuffer_time = max(delay - self.buffer_size, 0)

        # update video buffer
        self.buffer_size = max(self.buffer_size - delay, 0)
        self.buffer_size += 4.0  # each chunk is 4 seconds of video

        # bitrate change penalty
        if self.past_action is None:
            bitrate_change = 0
        else:
            bitrate_change = np.abs(self.bitrate_map[action] - \
                             self.bitrate_map[self.past_action])

        # linear reward
        # (https://dl.acm.org/citation.cfm?id=3098843 section 5.1, QoE metrics (1))
        reward = self.bitrate_map[action] - 4.3 * rebuffer_time - bitrate_change

        reward -= (self.buffer_size - 40) * (ACTION_DIM - action - 1)  # 时间充裕但不选择最大码率的惩罚

        # cap the buffer size
        self.buffer_size = min(self.buffer_size, 40.0)

        # store action for future bitrate change penalty
        self.past_action = action

        # update observed network bandwidth and duration
        self.past_chunk_throughputs.append(
            self.chunk_sizes[action][self.chunk_idx] / float(delay))
        self.past_chunk_download_times.append(delay)

        # Advance video
        self.chunk_idx += 1
        done = (self.chunk_idx == self.total_num_chunks)
        self.past_observation = self.observe()

        return self.past_observation if not self.normalize_obs else self.state_normalizer.normalize(self.past_observation), reward, done, \
               {
                    'bitrate': self.bitrate_map[action],
                    'rebuffer_time': rebuffer_time,
                    'buffer_size': self.buffer_size,
                   'chunk_idx': self.chunk_idx
                }

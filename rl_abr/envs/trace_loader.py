import os
import wget
import zipfile
import numpy as np
import rl_abr

def get_chunk_time(trace, t_idx):
    if t_idx == len(trace[0]) - 1:
        return 1  # bandwidth last for 1 second
    else:
        return trace[0][t_idx + 1] - trace[0][t_idx]

root_folder = '/envs/'

def load_chunk_sizes():
    # bytes of video chunk file at different bitrates

    # source video: "Envivio-Dash3" video H.264/MPEG-4 codec
    # at bitrates in {300,750,1200,1850,2850,4300} kbps

    # original video file:
    # https://github.com/hongzimao/pensieve/tree/master/video_server

    # download video size folder if not existed
    video_folder = rl_abr.__path__[0] + root_folder + 'videos/'
    #video_folder = rl_abr.__path__[0] + '/envs/abr_sim/videos/'

    create_folder_if_not_exists(video_folder)
    if not os.path.exists(video_folder + 'video_sizes.npy'):
        wget.download(
            'https://www.dropbox.com/s/hg8k8qq366y3u0d/video_sizes.npy?dl=1',
            out=video_folder + 'video_sizes.npy')

    chunk_sizes = np.load(video_folder + 'video_sizes.npy')

    return chunk_sizes


def load_traces(trace_type):
    assert(trace_type in ["n_train", "n_test"])
    #print(rl_abr.__path__)
    #print("0", rl_abr.__path__[0])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print("dir_path", dir_path)
    # download video size folder if not existed
    trace_folder = rl_abr.__path__[0] + root_folder + 'traces/' + trace_type + '/'
    create_folder_if_not_exists(trace_folder)
    print(trace_folder)
    print('rl_abr', rl_abr.__path__[0])
    #trace_folder = rl_abr.__path__[0] + '/envs/abr_sim/traces/'
    if trace_type == "n_train":
        if not os.path.exists(trace_folder):
            wget.download(
                'https://www.dropbox.com/s/xdlvykz9puhg5xd/cellular_traces.zip?dl=1',
                out=rl_abr.__path__[0] + root_folder)
            with zipfile.ZipFile(
                 rl_abr.__path__[0] + root_folder + 'cellular_traces.zip', 'r') as zip_f:
                zip_f.extractall(trace_folder)

        all_traces = []

        for trace in os.listdir(trace_folder):

            all_t = []
            all_bandwidth = []

            with open(trace_folder + trace, 'rb') as f:
                for line in f:
                    parse = line.split()
                    all_t.append(float(parse[0]))
                    all_bandwidth.append(float(parse[1]))

            all_traces.append((all_t, all_bandwidth))
    elif trace_type == "n_test":
        if not os.path.exists(trace_folder):
            wget.download(
                'https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAAD18W9IqDuLjocN7cvcpwCa/test_sim_traces?dl=1',
                out=rl_abr.__path__[0] + root_folder)
            with zipfile.ZipFile(
                 rl_abr.__path__[0] + root_folder + 'test_sim_traces.zip', 'r') as zip_f:
                zip_f.extractall(rl_abr.__path__[0] + root_folder)

        all_traces = []

        for trace in os.listdir(trace_folder):

            all_t = []
            all_bandwidth = []

            with open(trace_folder + trace, 'rb') as f:
                for line in f:
                    parse = line.split()
                    all_t.append(float(parse[0]))
                    all_bandwidth.append(float(parse[1]))

            all_traces.append((all_t, all_bandwidth))
    print("All traces len", len(all_traces))
    return all_traces


def sample_trace(all_traces, np_random):
    # weighted random sample based on trace length
    all_p = [len(trace[1]) for trace in all_traces]
    sum_p = float(sum(all_p))
    all_p = [p / sum_p for p in all_p]
    # sample a trace
    trace_idx = np_random.choice(len(all_traces), p=all_p)
    # sample a starting point
    init_t_idx = np_random.choice(len(all_traces[trace_idx][0]))
    # return a trace and the starting t
    return all_traces[trace_idx], init_t_idx


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

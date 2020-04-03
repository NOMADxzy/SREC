import os
import wget
import zipfile
import numpy as np
import pandas as pd
import rl_abr

def load_traces(trace, cache_size, rnd):
    trace_folder = rl_abr.__path__[0] + '/cache/traces/'
    if trace == 'test':
        if not os.path.exists(trace_folder):
            os.mkdir(trace_folder)
        if not os.path.exists(trace_folder + 'test_trace/'):
            wget.download(
                'https://www.dropbox.com/s/bfed1jk38sfvpez/test_trace.zip?dl=1',
                out=trace_folder)
            with zipfile.ZipFile(
                    trace_folder + 'test_trace.zip', 'r') as zip_f:
                zip_f.extractall(trace_folder)

        print('Load #%i trace for cache size of %i' % (rnd, cache_size))

        # load time, request id, request size
        df = pd.read_csv(trace_folder + 'test_trace/test_' + str(rnd) + '.tr', sep=' ', header=None)
        # remaining cache size, object last access time
        df[3], df[4] = cache_size, 0

    elif trace == 'real':
        if not os.path.exists(trace_folder):
            os.mkdir(trace_folder)
        if not os.path.exists(trace_folder + 'real_trace'):
            admission = input(
                "You are going to download a 3.0GB lzma file, are you sure you want to continue downloading? [Y/N]")
            if admission == 'Y':
                wget.download("http://dat-berger.de/cachetraces/sigmetrics18/cdn1_500m_sigmetrics18.tr.lzma",
                              out=trace_folder)
                os.system("xz --format=lzma --decompress " + trace_folder + "cdn1_500m_sigmetrics18.tr.lzma")
                os.system("mv " + trace_folder + "cdn1_500m_sigmetrics18.tr " + trace_folder + "real_trace")
            if admission == 'N':
                print(
                    "You can also manually download it by using this url:http://dat-berger.de/cachetraces/sigmetrics18/cdn1_500m_sigmetrics18.tr.lzma. Please unlzma it and rename it as real_trace.")
                exit(0)
        print('Load real trace for cache size of %i' % cache_size)
        df = pd.read_csv(trace_folder + 'real_trace', sep=' ', header=None)
        df[3], df[4] = cache_size, 0
    else:
        # load user's trace
        df = pd.read_csv(trace, sep=' ', header=None)
        df[3], df[4] = cache_size, 0

    return df
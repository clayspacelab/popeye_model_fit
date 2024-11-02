import numpy as np
from tqdm import tqdm
import cupy as cp
from itertools import product
from scipy.signal import fftconvolve
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, shared_memory
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize, NonlinearConstraint
import numba, time, ctypes
from numba import cuda, float32

# from cupyx.scipy.signal import fftconvolve
# from cupyx.scipy.stats import linregress
# import torch
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries, generate_rf_timeseries_nomask
import popeye.utilities_cclab as utils


from fit_utils import *


def test_cuda2(x, y, Nouter, use_gpu=False):
    if use_gpu:
        x = cp.array(x)
        y = cp.array(y)
        holder = cp.zeros(x.shape)
        for i in range(Nouter):
            holder += cp.dot(x, y)
        return holder
    else:
        holder = np.zeros(x.shape)
        for i in range(Nouter):
            holder += np.dot(x, y)
        return holder

def computeDot(args):
    x, y, use_gpu = args
    if use_gpu:
        x = cp.array(x)
        y = cp.array(y)
        return cp.dot(x, y)
    else:
        return np.dot(x, y)
    
def test_cuda(x, y, Nouter, use_gpu=False):
    args = [(x, y, use_gpu) for i in range(Nouter)]
    if use_gpu:
        # holder = cp.zeros(x.shape)
        with Pool(cpu_count()) as p:
            results = []
            for result in p.map(computeDot, args):
                results.append(result)
        holder = cp.sum(results)
            # holder = np.sum(p.map(computeDot, args))
        return holder
    else:
        with Pool(cpu_count()) as p:
            results = []
            for result in p.map(computeDot, args):
                results.append(result)
        holder = np.sum(results)
        # holder = np.zeros(x.shape)
        # with Pool(cpu_count()) as p:
        #     holder = np.sum(p.map(computeDot, args))
        return holder

        

# test_cuda(np.random.rand(10), np.random.rand(10))
startTime = time.time()
NInner = 3000
NOuter = 100
test_cuda(np.random.rand(NInner, NInner), np.random.rand(NInner, NInner), NOuter, use_gpu=False)
print('Time taken without GPU:', time.time() - startTime)

startTime = time.time()
test_cuda(np.random.rand(NInner, NInner), np.random.rand(NInner, NInner), NOuter, use_gpu=True)
print('Time taken with GPU:', time.time() - startTime)
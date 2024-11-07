import numpy as np
import cupy as cp
from multiprocessing import Pool, cpu_count
import time


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
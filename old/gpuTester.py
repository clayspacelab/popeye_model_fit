import numpy as np
from joblib import Parallel, delayed
import cupy as cp
import time
from numba import jit


@jit(nopython=True)
def cost_function(x, y):
    # Example of a simple quadratic function with a minimum at (3, -2)
    return (x - 3) ** 2 + (y + 2) ** 2

# Grid search parameters
nsize = 10000
grid_x = np.linspace(-10, 10, nsize)
grid_y = np.linspace(-10, 10, nsize)

# -----------------
# 1. CPU Parallelization
# -----------------
def cpu_grid_search():
    start = time.time()
    results = Parallel(n_jobs=-1)(delayed(cost_function)(x, y) for x in grid_x for y in grid_y)
    results = np.array(results).reshape(len(grid_x), len(grid_y))
    min_index = np.unravel_index(np.argmin(results), results.shape)
    min_x, min_y = grid_x[min_index[0]], grid_y[min_index[1]]
    end = time.time()
    print(f"CPU Minimum value found at (x, y) = ({min_x}, {min_y}) with cost = {results[min_index]}")
    print(f"CPU Grid Search Time: {end - start:.2f} seconds")

# -----------------
# 2. GPU Parallelization with CuPy
# -----------------
def gpu_grid_search():
    start = time.time()
    # Move the grid to the GPU
    grid_x_gpu = cp.linspace(-10, 10, nsize)
    grid_y_gpu = cp.linspace(-10, 10, nsize)
    x_gpu, y_gpu = cp.meshgrid(grid_x_gpu, grid_y_gpu)
    # Evaluate the cost function on the GPU
    results_gpu = (x_gpu - 3) ** 2 + (y_gpu + 2) ** 2
    # Find the minimum value and index
    min_index_gpu = cp.unravel_index(cp.argmin(results_gpu), results_gpu.shape)
    min_x_gpu, min_y_gpu = grid_x_gpu[min_index_gpu[0]], grid_y_gpu[min_index_gpu[1]]
    end = time.time()
    print(f"GPU Minimum value found at (x, y) = ({min_x_gpu}, {min_y_gpu}) with cost = {results_gpu[min_index_gpu]}")
    print(f"GPU Grid Search Time: {end - start:.2f} seconds")

# Run CPU and MPS tests
cpu_grid_search()
gpu_grid_search()

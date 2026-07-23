import numpy as np
from tqdm import tqdm
# import cupy as cp
from itertools import product
from scipy.signal import fftconvolve
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count, shared_memory
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize, NonlinearConstraint
import numba, time, ctypes
from numba import cuda

from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries, generate_rf_timeseries_nomask
import popeye.utilities_cclab as utils

from fit_utils import *


def generate_grid_prediction(args):
    # stimulus, x, y, sigma, n = args
    x, y, sigma, n, stimulus = args
    # Generate RF
    rf = generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
    # rf = utils.generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
    rf /= ((2 * np.pi * sigma**2) * 1/np.diff(stimulus.deg_x[0,0:2])**2)

    # Extract the stimulus time-series
    response = generate_rf_timeseries_nomask(stimulus.stim_arr, rf)
    # response = utils.generate_rf_timeseries(stimulus.stim_arr, rf)
    response **= n
    predsig = fftconvolve(response, utils.double_gamma_hrf(0, 1.3))[0:len(response)]
    # Normalize the units
    predsig = (predsig - np.mean(predsig)) / np.mean(predsig)

    return predsig

def getGridPreds(grid_space, stimulus, gridPath, nTRs):
    grid_preds = np.empty((len(grid_space), nTRs))#, dtype='float16')
    print("Starting prediction generation")
    with Pool(cpu_count()) as pool:
        results = pool.map(generate_grid_prediction, [(x, y, s, n, stimulus) for x, y, s, n in grid_space])

    for i, prediction in enumerate(results):
        grid_preds[i] = prediction
    # Save grid_preds to disk
    np.save(gridPath, grid_preds)
    return grid_preds

# @numba.jit(nopython=True, parallel=True)
def overload_estimate(estimate, data, prediction, use_gpu=False):
    # Returns (theta, r2, rho, sigma, n, x, y, beta, baseline)
    if use_gpu:
        # Raise error that GPU is not supported
        print("GPU not supported for this function. Please set use_gpu=False")
    else:
        X = np.vstack((np.ones(len(prediction)), prediction)).T
        XtX = np.dot(X.T, X)
        XtY = np.dot(X.T, data)
        betas = np.linalg.solve(XtX, XtY)
        scaled_prediction = np.dot(X, betas)
        r2 = np.corrcoef(data, scaled_prediction)[0, 1]**2
        theta = np.mod(np.arctan2(estimate[1], estimate[0]), 2*np.pi)
        rho = np.sqrt(estimate[0]**2 + estimate[1]**2)
    
        return (theta, r2, rho, estimate[2], estimate[3], estimate[0], estimate[1], betas[1], betas[0])

def compute_rmse(args):
    data, predictor_series, use_gpu = args
    predictor_series = predictor_series.reshape(-1, 1)
    y = data
    if use_gpu:
        # Raise error that GPU is not supported
        print("GPU not supported for this function. Please set use_gpu=False")
    else:
        X = np.hstack((np.ones((predictor_series.shape[0], 1)), predictor_series))
        XtX = np.dot(X.T, X)
        XtX_inv = np.linalg.inv(XtX)
        XtX_inv_Xt = np.dot(XtX_inv, X.T)
        betas = np.dot(XtX_inv_Xt, y)

        predictions = np.dot(X, betas)

        rmse = np.mean((data - predictions)**2)        
        return rmse

def process_voxel(args):
    timeseries_data, grid_preds, grid_space, use_gpu = args
    ngrids = len(grid_preds)
    
    args = [(timeseries_data, grid_preds[j], use_gpu) for j in range(ngrids)]

    if use_gpu:
        # Raise error that GPU is not supported
        print("GPU not supported for this function. Please set use_gpu=False")
    else:
        rmses = np.array([compute_rmse(arg) for arg in args])
        best_grid_idx = np.argmin(rmses)
        best_grid_estim = grid_space[best_grid_idx]
        best_grid_pred = grid_preds[best_grid_idx]
    overload_estim = overload_estimate(best_grid_estim, timeseries_data, best_grid_pred, use_gpu)
    
    return overload_estim

def get_grid_estims(grid_preds, grid_space, timeseries_data, gFit, indices, use_gpu=False, batch_size=1000):
    """
    Parallel grid fitting with optimized batch processing for surface vertices.
    
    Args:
        grid_preds: Pre-computed grid predictions
        grid_space: Grid parameter space
        timeseries_data: Vertex time series data
        gFit: Output array for grid fit results
        indices: Vertex indices
        use_gpu: Whether to use GPU acceleration
        batch_size: Number of vertices to process in each batch
    """
    nvoxs = len(timeseries_data)
    
    if use_gpu:
        # GPU implementation with CuPy
        try:
            import cupy as cp
            return get_grid_estims_gpu(grid_preds, grid_space, timeseries_data, gFit, indices, batch_size)
        except ImportError:
            print("CuPy not available. Falling back to CPU implementation.")
            use_gpu = False
    
    if not use_gpu:
        # CPU implementation with batch processing
        return get_grid_estims_cpu_batch(grid_preds, grid_space, timeseries_data, gFit, indices, batch_size)

def get_grid_estims_cpu_batch(grid_preds, grid_space, timeseries_data, gFit, indices, batch_size=1000):
    """CPU implementation with optimized batch processing."""
    nvoxs = len(timeseries_data)
    
    # Use shared memory for large arrays
    timeseries_data = utils.generate_shared_array(timeseries_data, ctypes.c_double)
    grid_preds = utils.generate_shared_array(grid_preds, ctypes.c_double)
    
    # Process in batches to manage memory
    start = 0
    pbar = tqdm(total=nvoxs, desc="Processing vertex batches", dynamic_ncols=True)
    
    while start < nvoxs:
        end = min(start + batch_size, nvoxs)
        batch_args = [
            (timeseries_data[iin, :], grid_preds, grid_space, False) 
            for iin in range(start, end)
        ]
        
        with Pool(cpu_count()) as pool:
            batch_results = list(pool.imap(process_voxel, batch_args))
        
        # Store results
        for idx, result in enumerate(batch_results):
            gFit[indices[start + idx], :] = result
        
        start = end
        pbar.update(end - start)
    
    pbar.close()
    return gFit

def get_grid_estims_gpu(grid_preds, grid_space, timeseries_data, gFit, indices, batch_size=2000):
    """
    Optimized GPU implementation with fully vectorized parallel processing.
    
    Key optimizations:
    1. All vertex-grid combinations computed in parallel
    2. Vectorized matrix operations across entire batches
    3. Optimized memory usage with larger batch sizes
    4. Minimal CPU-GPU transfers
    """
    import cupy as cp
    
    nvoxs = len(timeseries_data)
    ngrids = len(grid_preds)
    
    print(f"GPU Processing: {nvoxs} vertices, {ngrids} grids, batch_size={batch_size}")
    
    # Move data to GPU once
    timeseries_gpu = cp.asarray(timeseries_data, dtype=cp.float32)
    grid_preds_gpu = cp.asarray(grid_preds, dtype=cp.float32)
    
    # Process in optimized batches
    start = 0
    pbar = tqdm(total=nvoxs, desc="GPU Vectorized Processing", dynamic_ncols=True)
    
    while start < nvoxs:
        end = min(start + batch_size, nvoxs)
        batch_timeseries = timeseries_gpu[start:end]
        
        # Fully vectorized computation for all vertices in batch
        batch_results = process_vertex_batch_gpu(
            batch_timeseries, grid_preds_gpu, grid_space
        )
        
        # Store results efficiently
        for idx, result in enumerate(batch_results):
            gFit[indices[start + idx], :] = result
        
        start = end
        pbar.update(end - start)
    
    pbar.close()
    
    # Clear GPU memory
    del timeseries_gpu, grid_preds_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return gFit

def process_vertex_batch_gpu(timeseries_batch, grid_preds_gpu, grid_space):
    """Process a batch of vertices on GPU using fully vectorized operations."""
    import cupy as cp
    
    batch_size, nTRs = timeseries_batch.shape
    ngrids = len(grid_preds_gpu)
    
    # Fully vectorized computation across all vertices and grids
    # Shape: (batch_size, ngrids, nTRs)
    timeseries_expanded = cp.expand_dims(timeseries_batch, axis=1)  # (batch_size, 1, nTRs)
    grid_preds_expanded = cp.expand_dims(grid_preds_gpu, axis=0)    # (1, ngrids, nTRs)
    
    # Compute RMSE for all vertex-grid combinations simultaneously
    # This is the key optimization - process all combinations in parallel
    rmses = compute_rmse_vectorized_gpu(timeseries_expanded, grid_preds_expanded)
    
    # Find best grid for each vertex
    best_grid_indices = cp.argmin(rmses, axis=1)  # (batch_size,)
    
    # Extract best results for each vertex
    results = []
    for i in range(batch_size):
        best_idx = int(best_grid_indices[i])
        best_grid_estim = grid_space[best_idx]
        best_grid_pred = grid_preds_gpu[best_idx]
        vertex_data = timeseries_batch[i]
        
        # Compute final estimate
        result = overload_estimate_gpu(best_grid_estim, vertex_data, best_grid_pred)
        results.append(result)
    
    return results

def compute_rmse_vectorized_gpu(timeseries_expanded, grid_preds_expanded):
    """
    Vectorized RMSE computation across all vertex-grid combinations.
    
    Args:
        timeseries_expanded: (batch_size, 1, nTRs)
        grid_preds_expanded: (1, ngrids, nTRs)
    
    Returns:
        rmses: (batch_size, ngrids) - RMSE for each vertex-grid combination
    """
    import cupy as cp
    
    batch_size, _, nTRs = timeseries_expanded.shape
    _, ngrids, _ = grid_preds_expanded.shape
    
    # Create design matrix for all combinations
    # Shape: (batch_size, ngrids, nTRs, 2) - last dim is [ones, predictors]
    ones = cp.ones((batch_size, ngrids, nTRs, 1))
    predictors = grid_preds_expanded[..., cp.newaxis]  # (1, ngrids, nTRs, 1)
    
    # Broadcast to all combinations
    X = cp.concatenate([ones, predictors], axis=-1)  # (batch_size, ngrids, nTRs, 2)
    
    # Reshape for batch matrix operations
    X_flat = X.reshape(-1, nTRs, 2)  # (batch_size * ngrids, nTRs, 2)
    y_flat = timeseries_expanded.reshape(-1, nTRs)  # (batch_size * ngrids, nTRs)
    
    # Vectorized least squares for all combinations
    XtX = cp.sum(X_flat[:, :, :, cp.newaxis] * X_flat[:, :, cp.newaxis, :], axis=1)  # (batch_size * ngrids, 2, 2)
    XtY = cp.sum(X_flat * y_flat[:, :, cp.newaxis], axis=1)  # (batch_size * ngrids, 2)
    
    # Solve for betas
    betas = cp.linalg.solve(XtX, XtY)  # (batch_size * ngrids, 2)
    
    # Compute predictions
    predictions = cp.sum(X_flat * betas[:, cp.newaxis, :], axis=2)  # (batch_size * ngrids, nTRs)
    
    # Compute RMSE
    rmse_flat = cp.mean((y_flat - predictions)**2, axis=1)  # (batch_size * ngrids,)
    
    # Reshape back to (batch_size, ngrids)
    rmses = rmse_flat.reshape(batch_size, ngrids)
    
    return rmses

def overload_estimate_gpu(estimate, data, prediction):
    """GPU version of overload_estimate using CuPy."""
    import cupy as cp
    
    X = cp.vstack((cp.ones(len(prediction)), prediction)).T
    XtX = cp.dot(X.T, X)
    XtY = cp.dot(X.T, data)
    betas = cp.linalg.solve(XtX, XtY)
    scaled_prediction = cp.dot(X, betas)
    r2 = cp.corrcoef(data, scaled_prediction)[0, 1]**2
    theta = cp.mod(cp.arctan2(estimate[1], estimate[0]), 2*cp.pi)
    rho = cp.sqrt(estimate[0]**2 + estimate[1]**2)
    
    return (theta, r2, rho, estimate[2], estimate[3], estimate[0], estimate[1], betas[1], betas[0])

def get_grid2_estims(grid_preds, grid_space, voxel_data):
    """Optimized grid estimation for single voxel."""
    ngrids = len(grid_preds)
    rmses = []
    for j in range(ngrids):
        rmses.append(compute_rmse([voxel_data, grid_preds[j]]))
    best_grid_estim = grid_space[np.argmin(rmses)]
    overload_estim = overload_estimate(best_grid_estim, voxel_data, grid_preds[np.argmin(rmses)])
    return overload_estim

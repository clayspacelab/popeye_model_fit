"""
H05_grid_fit.py — Grid fitting: find the best-matching grid prediction per voxel/vertex.

For each voxel/vertex, computes RMSE against all grid predictions and selects
the best match. Then computes the full overload estimate (R², beta, baseline)
via OLS regression.

Supports both CPU (default, multiprocessing) and GPU (optional, CuPy) paths.

Key functions:
    overload_estimate()  — OLS regression to get beta, baseline, R²
    compute_rmse()       — RMSE between data and one grid prediction
    process_voxel()      — Find best grid match for one voxel/vertex
    get_grid_estims()    — Parallel grid fitting across all voxels/vertices
"""

import numpy as np
import ctypes
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import popeye.utilities_cclab as utils


# ---------------------------------------------------------------------------
# Core computation functions (CPU)
# ---------------------------------------------------------------------------

def overload_estimate(estimate, data, prediction, use_gpu=False):
    """
    Compute the full pRF estimate via OLS regression.

    Given a grid parameter estimate and its prediction, fit beta and baseline
    via ordinary least squares, then compute R² and polar coordinates.

    Parameters
    ----------
    estimate : array-like
        Grid parameters (x, y, sigma, n) or similar.
    data : ndarray
        Observed BOLD timeseries.
    prediction : ndarray
        Model-predicted timeseries.
    use_gpu : bool
        If True, use CuPy for GPU acceleration.

    Returns
    -------
    tuple of 9 floats
        (theta, r2, rho, sigma, n, x, y, beta, baseline)
    """
    if use_gpu:
        return _overload_estimate_gpu(estimate, data, prediction)

    X = np.vstack((np.ones(len(prediction)), prediction)).T
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, data)
    betas = np.linalg.solve(XtX, XtY)
    scaled_prediction = np.dot(X, betas)
    r2 = np.corrcoef(data, scaled_prediction)[0, 1]**2
    theta = np.mod(np.arctan2(estimate[1], estimate[0]), 2 * np.pi)
    rho = np.sqrt(estimate[0]**2 + estimate[1]**2)

    return (theta, r2, rho, estimate[2], estimate[3],
            estimate[0], estimate[1], betas[1], betas[0])


def compute_rmse(args):
    """
    Compute RMSE between observed data and a grid prediction via OLS.

    Parameters
    ----------
    args : tuple
        (data, predictor_series, use_gpu)

    Returns
    -------
    float
        Mean squared error.
    """
    data, predictor_series, use_gpu = args

    if use_gpu:
        return _compute_rmse_gpu(data, predictor_series)

    predictor_series = predictor_series.reshape(-1, 1)
    X = np.hstack((np.ones((predictor_series.shape[0], 1)), predictor_series))
    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.inv(XtX)
    XtX_inv_Xt = np.dot(XtX_inv, X.T)
    betas = np.dot(XtX_inv_Xt, data)
    predictions = np.dot(X, betas)
    rmse = np.mean((data - predictions)**2)
    return rmse


def process_voxel(args):
    """
    Find the best-matching grid prediction for a single voxel/vertex.

    Parameters
    ----------
    args : tuple
        (timeseries_data, grid_preds, grid_space, use_gpu)

    Returns
    -------
    tuple of 9 floats
        Overload estimate for this voxel/vertex.
    """
    timeseries_data, grid_preds, grid_space, use_gpu = args
    ngrids = len(grid_preds)

    rmse_args = [(timeseries_data, grid_preds[j], use_gpu) for j in range(ngrids)]
    rmses = np.array([compute_rmse(arg) for arg in rmse_args])

    best_grid_idx = np.argmin(rmses)
    best_grid_estim = grid_space[best_grid_idx]
    best_grid_pred = grid_preds[best_grid_idx]

    overload_estim = overload_estimate(
        best_grid_estim, timeseries_data, best_grid_pred, use_gpu
    )
    return overload_estim


# ---------------------------------------------------------------------------
# Main grid fitting function
# ---------------------------------------------------------------------------

def get_grid_estims(grid_preds, grid_space, timeseries_data, gFit, indices,
                    use_gpu=False, batch_size=1000):
    """
    Find the best grid match for all voxels/vertices.

    Parameters
    ----------
    grid_preds : ndarray
        Pre-computed grid predictions (n_grids, n_timepoints).
    grid_space : list
        Grid parameter space, aligned with grid_preds.
    timeseries_data : ndarray
        Observed data (n_voxels, n_timepoints).
    gFit : ndarray
        Output array to fill with grid fit results.
    indices : list
        Indices into gFit for each voxel/vertex.
        - Volumetric: list of (x, y, z) tuples → gFit[x, y, z, :]
        - Surface: list of int → gFit[idx, :]
    use_gpu : bool
        If True, use GPU-accelerated path.
    batch_size : int
        Batch size for GPU processing. Ignored for CPU.

    Returns
    -------
    gFit : ndarray
        Updated grid fit array.
    """
    nvoxs = len(timeseries_data)

    if use_gpu:
        try:
            import cupy as cp
            return _get_grid_estims_gpu(
                grid_preds, grid_space, timeseries_data, gFit,
                indices, batch_size
            )
        except ImportError:
            print("CuPy not available. Falling back to CPU implementation.")

    # --- CPU path: shared memory + multiprocessing ---
    timeseries_data = utils.generate_shared_array(timeseries_data, ctypes.c_double)
    grid_preds = utils.generate_shared_array(grid_preds, ctypes.c_double)

    args = [(timeseries_data[iin, :], grid_preds, grid_space, False)
            for iin in range(nvoxs)]

    with Pool(cpu_count()) as pool:
        results = []
        for result in tqdm(pool.imap(process_voxel, args),
                           total=nvoxs, dynamic_ncols=False):
            results.append(result)

    for i, result in enumerate(results):
        idx = indices[i]
        if isinstance(idx, (list, tuple)):
            gFit[idx[0], idx[1], idx[2], :] = result  # volumetric 3D index
        else:
            gFit[idx, :] = result  # surface 1D index

    return gFit


# ---------------------------------------------------------------------------
# GPU implementations (optional, requires CuPy)
# ---------------------------------------------------------------------------

def _overload_estimate_gpu(estimate, data, prediction):
    """GPU version of overload_estimate using CuPy."""
    import cupy as cp

    X = cp.vstack((cp.ones(len(prediction)), prediction)).T
    XtX = cp.dot(X.T, X)
    XtY = cp.dot(X.T, data)
    betas = cp.linalg.solve(XtX, XtY)
    scaled_prediction = cp.dot(X, betas)
    r2 = cp.corrcoef(data, scaled_prediction)[0, 1]**2
    theta = cp.mod(cp.arctan2(estimate[1], estimate[0]), 2 * cp.pi)
    rho = cp.sqrt(estimate[0]**2 + estimate[1]**2)

    return (theta, r2, rho, estimate[2], estimate[3],
            estimate[0], estimate[1], betas[1], betas[0])


def _compute_rmse_gpu(data, predictor_series):
    """GPU version of compute_rmse using CuPy."""
    import cupy as cp

    predictor_series = predictor_series.reshape(-1, 1)
    X = cp.hstack((cp.ones((predictor_series.shape[0], 1)), predictor_series))
    XtX = cp.dot(X.T, X)
    XtX_inv = cp.linalg.inv(XtX)
    XtX_inv_Xt = cp.dot(XtX_inv, X.T)
    betas = cp.dot(XtX_inv_Xt, data)
    predictions = cp.dot(X, betas)
    rmse = cp.mean((data - predictions)**2)
    return rmse


def _get_grid_estims_gpu(grid_preds, grid_space, timeseries_data, gFit,
                         indices, batch_size=2000):
    """
    GPU-accelerated grid fitting with fully vectorized batch processing.

    Optimizations:
        1. All vertex-grid combinations computed in parallel
        2. Vectorized matrix operations across entire batches
        3. Minimal CPU-GPU transfers
    """
    import cupy as cp

    nvoxs = len(timeseries_data)
    ngrids = len(grid_preds)

    print(f"GPU Processing: {nvoxs} vertices, {ngrids} grids, batch_size={batch_size}")

    # Move data to GPU once
    timeseries_gpu = cp.asarray(timeseries_data, dtype=cp.float32)
    grid_preds_gpu = cp.asarray(grid_preds, dtype=cp.float32)

    # Process in batches
    start = 0
    pbar = tqdm(total=nvoxs, desc="GPU Vectorized Processing", dynamic_ncols=True)

    while start < nvoxs:
        end = min(start + batch_size, nvoxs)
        batch_timeseries = timeseries_gpu[start:end]

        batch_results = _process_vertex_batch_gpu(
            batch_timeseries, grid_preds_gpu, grid_space
        )

        for idx_offset, result in enumerate(batch_results):
            idx = indices[start + idx_offset]
            if isinstance(idx, (list, tuple)):
                gFit[idx[0], idx[1], idx[2], :] = result
            else:
                gFit[idx, :] = result

        pbar.update(end - start)
        start = end

    pbar.close()

    # Clean GPU memory
    del timeseries_gpu, grid_preds_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return gFit


def _process_vertex_batch_gpu(timeseries_batch, grid_preds_gpu, grid_space):
    """Process a batch of vertices on GPU using vectorized operations."""
    import cupy as cp

    batch_size, nTRs = timeseries_batch.shape
    ngrids = len(grid_preds_gpu)

    # Expand for broadcasting: (batch, 1, nTRs) vs (1, ngrids, nTRs)
    timeseries_expanded = cp.expand_dims(timeseries_batch, axis=1)
    grid_preds_expanded = cp.expand_dims(grid_preds_gpu, axis=0)

    # Vectorized RMSE computation
    rmses = _compute_rmse_vectorized_gpu(timeseries_expanded, grid_preds_expanded)

    # Find best grid for each vertex
    best_grid_indices = cp.argmin(rmses, axis=1)

    results = []
    for i in range(batch_size):
        best_idx = int(best_grid_indices[i])
        best_grid_estim = grid_space[best_idx]
        best_grid_pred = grid_preds_gpu[best_idx]
        vertex_data = timeseries_batch[i]
        result = _overload_estimate_gpu(best_grid_estim, vertex_data, best_grid_pred)
        results.append(result)

    return results


def _compute_rmse_vectorized_gpu(timeseries_expanded, grid_preds_expanded):
    """
    Vectorized RMSE across all vertex-grid combinations on GPU.

    Parameters
    ----------
    timeseries_expanded : cp.ndarray, shape (batch_size, 1, nTRs)
    grid_preds_expanded : cp.ndarray, shape (1, ngrids, nTRs)

    Returns
    -------
    rmses : cp.ndarray, shape (batch_size, ngrids)
    """
    import cupy as cp

    batch_size, _, nTRs = timeseries_expanded.shape
    _, ngrids, _ = grid_preds_expanded.shape

    ones = cp.ones((batch_size, ngrids, nTRs, 1))
    predictors = grid_preds_expanded[..., cp.newaxis]
    X = cp.concatenate([ones, predictors], axis=-1)

    X_flat = X.reshape(-1, nTRs, 2)
    y_flat = timeseries_expanded.reshape(-1, nTRs)

    # Vectorized least squares
    XtX = cp.sum(X_flat[:, :, :, cp.newaxis] * X_flat[:, :, cp.newaxis, :], axis=1)
    XtY = cp.sum(X_flat * y_flat[:, :, cp.newaxis], axis=1)
    betas = cp.linalg.solve(XtX, XtY)

    predictions = cp.sum(X_flat * betas[:, cp.newaxis, :], axis=2)
    rmse_flat = cp.mean((y_flat - predictions)**2, axis=1)
    rmses = rmse_flat.reshape(batch_size, ngrids)

    return rmses

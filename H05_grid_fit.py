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
                    use_gpu=False, batch_size=2000):
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

    return (float(theta.get()), float(r2.get()), float(rho.get()),
            float(estimate[2]), float(estimate[3]),
            float(estimate[0]), float(estimate[1]),
            float(betas[1].get()), float(betas[0].get()))


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
    return float(rmse.get())


def _get_grid_estims_gpu(grid_preds, grid_space, timeseries_data, gFit,
                         indices, batch_size=2000):
    """
    GPU-accelerated grid fitting via high-throughput cuBLAS matrix multiplication.

    Computes OLS regression fits for all voxel-grid combinations in parallel:
        Y (B x T) @ X^T (T x G) -> S_xy (B x G) matrix product on GPU.
    """
    import cupy as cp

    nvoxs = len(timeseries_data)
    ngrids = len(grid_preds)
    nTRs = timeseries_data.shape[1]

    print(f"GPU Processing: {nvoxs} voxels, {ngrids} grid points, batch_size={batch_size}")

    # Transfer data to GPU
    timeseries_gpu = cp.asarray(timeseries_data, dtype=cp.float32)  # (B, T)
    grid_preds_gpu = cp.asarray(grid_preds, dtype=cp.float32)       # (G, T)

    # Precompute grid summary statistics on GPU once
    grid_means = cp.mean(grid_preds_gpu, axis=1, keepdims=True)     # (G, 1)
    X_centered = grid_preds_gpu - grid_means                        # (G, T)
    S_xx = cp.sum(X_centered**2, axis=1)                            # (G,)
    S_xx[S_xx == 0] = 1e-8

    start = 0
    pbar = tqdm(total=nvoxs, desc="GPU Matrix Processing", dynamic_ncols=True)

    while start < nvoxs:
        end = min(start + batch_size, nvoxs)
        batch_y = timeseries_gpu[start:end]                         # (B_sub, T)
        b_sub = batch_y.shape[0]

        # Center batch voxels
        y_means = cp.mean(batch_y, axis=1, keepdims=True)            # (B_sub, 1)
        Y_centered = batch_y - y_means                              # (B_sub, T)
        S_yy = cp.sum(Y_centered**2, axis=1, keepdims=True)          # (B_sub, 1)

        # Matrix multiplication via cuBLAS: S_xy = Y_centered @ X_centered.T
        S_xy = cp.dot(Y_centered, X_centered.T)                     # (B_sub, G)

        # OLS slope: beta1 = S_xy / S_xx
        betas1 = S_xy / S_xx[cp.newaxis, :]                          # (B_sub, G)

        # SSE = S_yy - beta1 * S_xy
        sse = S_yy - (betas1 * S_xy)                                 # (B_sub, G)
        rmses = sse / nTRs

        # Mask out negative slope estimates (beta1 < 0 -> invalid pRF fit)
        rmses[betas1 < 0] = 1e9

        # Best grid index for each voxel in batch
        best_grid_indices = cp.argmin(rmses, axis=1)                 # (B_sub,)

        # Compute full overload estimate for best grid matches
        best_grid_indices_cpu = cp.asnumpy(best_grid_indices)
        for i in range(b_sub):
            best_idx = int(best_grid_indices_cpu[i])
            best_grid_estim = grid_space[best_idx]
            best_grid_pred = grid_preds_gpu[best_idx]
            voxel_data = batch_y[i]

            result = _overload_estimate_gpu(best_grid_estim, voxel_data, best_grid_pred)

            idx = indices[start + i]
            if isinstance(idx, (list, tuple)):
                gFit[idx[0], idx[1], idx[2], :] = result
            else:
                gFit[idx, :] = result

        start = end
        pbar.update(b_sub)

    pbar.close()

    # Free GPU memory
    del timeseries_gpu, grid_preds_gpu, X_centered, S_xx
    cp.get_default_memory_pool().free_all_blocks()

    return gFit

"""
H05_grid_fit.py — Grid fitting: find the best-matching grid prediction per voxel/vertex.

For each voxel/vertex, computes RMSE against all grid predictions and selects
the best match. Then computes the full overload estimate (R², beta, baseline)
via OLS regression.

Supports both CPU (default, multiprocessing) and GPU (optional, CuPy) paths.

CPU path uses fully vectorized OLS across all grid points at once per voxel
(single numpy matmul instead of a Python loop), which saturates each worker core.

Key functions:
    overload_estimate()  — OLS regression to get beta, baseline, R²
    process_voxel()      — Find best grid match for one voxel/vertex (vectorized)
    get_grid_estims()    — Parallel grid fitting across all voxels/vertices
"""

import numpy as np
import os
import jax
#jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_debug_infs', True)
#jax.config.update("jax_disable_jit", True)
#jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp


def overload_estimate_jax(estimate, data, prediction):
    """
    Compute the full pRF estimate via covariance trick equivalent to OLS regression.

    Given a grid parameter estimate and its prediction, fit beta and baseline
    via ordinary least squares, then compute R² and polar coordinates.

    Parameters
    ----------
    estimate : array-like
        Grid parameters (x, y, sigma, n) or similar.
    data : Array
        Observed BOLD timeseries.
    prediction : Array
        Model-predicted timeseries.

    Returns
    -------
    jnp Array of 9 floats
        (theta, r2, rho, sigma, n, x, y, beta, baseline)
    """
    #on z-score scale, beta is just the covariance of data and prediction.
    #since correlation is normalized covariance, square of beta is r2
    slope = jnp.dot(prediction, data) / prediction.shape[0]
    r2 = slope ** 2

    #this would be the way to get intercept, but w/ z-score any deviation from zero is numerical slop
    #so we set to zero below
    #intercept = jnp.mean(data) - slope*jnp.mean(prediction)

    theta = jnp.mod(jnp.arctan2(estimate[1], estimate[0]), 2 * jnp.pi)
    rho = jnp.sqrt(estimate[0] ** 2 + estimate[1] ** 2)
    return jnp.stack((theta, r2, rho, estimate[2], estimate[3],
                      estimate[0], estimate[1], slope,
                      jnp.asarray(0.0, dtype=estimate.dtype)))


def process_voxel_jax(y,grid_preds,grid_space):
    """
    Find the best-matching grid prediction for a single voxel/vertex.

    Uses fully vectorized OLS-like across all grid points at once:
      - Computes beta = grid_preds @ y_centered / T  in one matmul 
      - This works because beta = covariance for z-scored predictors
      - See: https://en.wikipedia.org/wiki/Simple_linear_regression#Relationship_with_the_sample_covariance_matrix
      - Masks negative-slope fits (invalid pRF response) before argmin
    This replaces the old serial Python loop over 77k grid points.


    Parameters
    ----------
    y : ndarray (T,)
        Observed timeseries for this voxel.

    Returns
    -------
    tuple of 9 floats
        Overload estimate for this voxel/vertex.
    """

    betas1 = (grid_preds @ y)/grid_preds.shape[1]
    #1-r**2 in liu of full sse (variance unexplained) since only differs from see by fixed scale factor per voxel
    vue = 1 - betas1**2

    # Mask invalid fits: negative slope = pRF predicts wrong sign
    vue = jnp.where(betas1 < 0, np.inf, vue)

    best_grid_idx = jnp.argmin(vue)
    best_grid_estim = grid_space[best_grid_idx]
    best_grid_pred = grid_preds[best_grid_idx]

    return overload_estimate_jax(best_grid_estim, y, best_grid_pred)


# ---------------------------------------------------------------------------
# Main grid fitting function
# ---------------------------------------------------------------------------

def get_grid_estims(grid_preds, grid_space, timeseries_data, gFit, indices,
                    use_gpu=False, batch_size=2000):
    """
    Find the best grid match for all voxels/vertices.

    CPU path precomputes centered grid statistics once, then dispatches
    one vectorized worker task per voxel. Each worker does a single matmul
    across all G grid points instead of a Python loop, fully saturating
    its assigned CPU core.

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

    #jitted/vectorized OLS across voxels/grid points, potentially batched to avoid memory overload
    @jax.jit(static_argnames='batch_size')
    def process_voxels(all_data,preds,grids,batch_size):

        def process_one(data):
            return process_voxel_jax(data,preds,grids)
        
        return jax.lax.map(process_one,all_data,batch_size=batch_size)

    results = process_voxels(jnp.asarray(timeseries_data),jnp.asarray(grid_preds),jnp.asarray(grid_space),batch_size)
    results = jax.device_get(results)

    for i, result in enumerate(results):
        idx = indices[i]
        if isinstance(idx, (list, tuple)):
            gFit[idx[0], idx[1], idx[2], :] = result  # volumetric 3D index
        else:
            gFit[idx, :] = result  # surface 1D index

    return gFit

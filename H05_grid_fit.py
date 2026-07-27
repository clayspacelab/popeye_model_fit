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
import ctypes
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import popeye.utilities_cclab as utils

# Module-level globals populated by the Pool worker initializer
_G_centered = None
_S_xx = None
_grid_space = None


def _worker_init(G_centered, S_xx, grid_space):
    """Initializer for each Pool worker: store shared arrays as globals."""
    global _G_centered, _S_xx, _grid_space
    _G_centered = G_centered
    _S_xx = S_xx
    _grid_space = grid_space


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


def process_voxel(y):
    """
    Find the best-matching grid prediction for a single voxel/vertex.

    Uses fully vectorized OLS across all grid points at once:
      - Centers the voxel timeseries
      - Computes S_xy = G_centered @ y_centered  in one matmul  (G,)
      - Derives OLS slope beta1 = S_xy / S_xx and SSE analytically
      - Masks negative-slope fits (invalid pRF response) before argmin
    This replaces the old serial Python loop over 77k grid points.

    G_centered, S_xx, and grid_space are set once per worker via the Pool
    initializer (_worker_init), so they are NOT pickled with every task.

    Parameters
    ----------
    y : ndarray (T,)
        Observed timeseries for this voxel.

    Returns
    -------
    tuple of 9 floats
        Overload estimate for this voxel/vertex.
    """
    G_centered = _G_centered
    S_xx = _S_xx
    grid_space = _grid_space

    # Center the voxel timeseries
    y_c = y - y.mean()                     # (T,)
    S_yy = float(y_c @ y_c)               # scalar

    # Vectorized OLS across all G grid points in one matmul
    S_xy = G_centered @ y_c               # (G,)  — the hot path
    betas1 = S_xy / S_xx                  # (G,)  OLS slope
    sse = S_yy - betas1 * S_xy            # (G,)  SSE

    # Mask invalid fits: negative slope = pRF predicts wrong sign
    sse[betas1 < 0] = np.inf

    best_grid_idx = int(np.argmin(sse))
    best_grid_estim = grid_space[best_grid_idx]
    best_grid_pred = G_centered[best_grid_idx]  # centered pred is fine for OLS

    return overload_estimate(best_grid_estim, y, best_grid_pred)


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

    # --- CPU path: vectorized OLS across all grid points per voxel ---
    # Precompute centered grid stats once — injected into each worker via initializer
    # so the ~62MB G_centered array is NOT pickled with every voxel task.
    grid_preds = np.asarray(grid_preds, dtype=np.float32)
    G_means = grid_preds.mean(axis=1, keepdims=True)   # (G, 1)
    G_centered = grid_preds - G_means                  # (G, T)
    S_xx = (G_centered ** 2).sum(axis=1)               # (G,)
    S_xx[S_xx == 0] = 1e-8                             # guard flat predictions

    timeseries_data = np.asarray(timeseries_data, dtype=np.float32)

    # Each task is just the voxel timeseries (201 floats, ~800 bytes)
    voxel_args = [timeseries_data[iin] for iin in range(nvoxs)]

    # chunksize: amortize IPC overhead across multiple voxels per round-trip
    n_workers = cpu_count()
    chunksize = max(1, nvoxs // (n_workers * 4))

    with Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(G_centered, S_xx, grid_space)
    ) as pool:
        results = list(tqdm(
            pool.imap(process_voxel, voxel_args, chunksize=chunksize),
            total=nvoxs, dynamic_ncols=True
        ))

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
    GPU-accelerated grid fitting with dynamic dual-tiled memory management.

    With large grids (e.g. Ns=100 -> 2.1M points), the S_xy result matrix
    (B_vox x G) is the dominant allocation. For G=2.1M and B_vox=2000:
        2000 x 2.1M x 4 bytes = 16.86 GB -- exceeds any consumer GPU.

    Fix: query free GPU memory at runtime and compute safe tile sizes for
    BOTH the voxel (B_vox) and grid (G_chunk) dimensions so peak usage
    stays within budget regardless of grid size.

    Strategy:
      - X_centered (G, T) and S_xx (G,) stay resident throughout
      - Voxels tiled into B_vox-sized batches
      - Grid tiled into G_chunk-sized chunks for the matmul
      - Running argmin tracks best-per-voxel without ever holding full (B, G)
    """
    import cupy as cp

    nvoxs  = len(timeseries_data)
    ngrids = len(grid_preds)
    nTRs   = timeseries_data.shape[1]

    # ── Query free memory and compute safe tile sizes ─────────────────────────
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

    # Resident arrays: X_centered (G, T) + S_xx (G,) + timeseries_gpu (N, T)
    resident_bytes = (ngrids * nTRs + ngrids + nvoxs * nTRs) * 4
    safety_margin  = 512 * 1024**2   # 512 MB headroom
    usable_bytes   = max(free_bytes - resident_bytes - safety_margin, 0)

    # Per-tile peak: S_xy (B, G_chunk) + beta1 (B, G_chunk) + sse (B, G_chunk)
    # = 3 * B * G_chunk * 4 bytes.  Budget: usable_bytes
    max_elements = usable_bytes // (3 * 4)

    # Choose G_chunk <= ngrids; B_vox fills remaining budget
    G_chunk = min(ngrids, max(256, int(max_elements ** 0.5)))
    B_vox   = min(nvoxs,  max(1,   int(max_elements // G_chunk)))

    print(f"GPU Grid Fit : {nvoxs:,} voxels | {ngrids:,} grid pts | "
          f"B_vox={B_vox} | G_chunk={G_chunk:,} | "
          f"free={free_bytes/1e9:.1f} GB / {total_bytes/1e9:.1f} GB")

    # ── Transfer to GPU ───────────────────────────────────────────────────────
    timeseries_gpu = cp.asarray(timeseries_data, dtype=cp.float32)  # (N, T)
    grid_preds_gpu = cp.asarray(grid_preds,      dtype=cp.float32)  # (G, T)

    grid_means = grid_preds_gpu.mean(axis=1, keepdims=True)
    X_centered = grid_preds_gpu - grid_means                        # (G, T)
    S_xx       = (X_centered ** 2).sum(axis=1)                     # (G,)
    S_xx[S_xx == 0] = 1e-8
    del grid_preds_gpu, grid_means

    # ── Dual-tiled fitting ────────────────────────────────────────────────────
    pbar = tqdm(total=nvoxs, desc="GPU Grid Fit", dynamic_ncols=True)

    for v_start in range(0, nvoxs, B_vox):
        v_end   = min(v_start + B_vox, nvoxs)
        batch_y = timeseries_gpu[v_start:v_end]              # (B, T)
        B       = batch_y.shape[0]

        y_mean    = batch_y.mean(axis=1, keepdims=True)
        Y_centered = batch_y - y_mean                         # (B, T)
        S_yy      = (Y_centered ** 2).sum(axis=1)            # (B,)

        # Running best across G_chunks
        best_sse = cp.full(B, cp.inf, dtype=cp.float32)
        best_idx = cp.zeros(B,       dtype=cp.int64)

        for g_start in range(0, ngrids, G_chunk):
            g_end     = min(g_start + G_chunk, ngrids)
            Xc_chunk  = X_centered[g_start:g_end]            # (Gc, T)
            Sxx_chunk = S_xx[g_start:g_end]                  # (Gc,)

            # (B, T) @ (T, Gc) -> (B, Gc)
            S_xy  = Y_centered @ Xc_chunk.T
            beta1 = S_xy / Sxx_chunk[cp.newaxis, :]
            sse   = S_yy[:, None] - beta1 * S_xy

            # Invalid fits: negative slope
            sse[beta1 < 0] = cp.inf

            # Update running argmin
            chunk_best_sse = sse.min(axis=1)                 # (B,)
            chunk_best_idx = sse.argmin(axis=1)              # (B,) local
            improved       = chunk_best_sse < best_sse
            best_sse       = cp.where(improved, chunk_best_sse, best_sse)
            best_idx       = cp.where(improved,
                                      chunk_best_idx + g_start,
                                      best_idx)

            del S_xy, beta1, sse, Xc_chunk, Sxx_chunk

        # Overload estimate for each voxel in batch
        best_idx_cpu = cp.asnumpy(best_idx)
        for i in range(B):
            b_idx  = int(best_idx_cpu[i])
            result = _overload_estimate_gpu(
                grid_space[b_idx],
                batch_y[i],
                X_centered[b_idx],   # centered pred; overload_estimate re-fits OLS
            )
            idx = indices[v_start + i]
            if isinstance(idx, (list, tuple)):
                gFit[idx[0], idx[1], idx[2], :] = result
            else:
                gFit[idx, :] = result

        del Y_centered, best_sse, best_idx
        pbar.update(B)

    pbar.close()

    del timeseries_gpu, X_centered, S_xx
    cp.get_default_memory_pool().free_all_blocks()

    return gFit

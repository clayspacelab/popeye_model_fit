"""
H06_final_fit.py — Gradient-descent refinement of pRF estimates.

CPU path: parallel L-BFGS-B per voxel via multiprocessing Pool.
  - Pool initializer injects the stimulus once per worker (avoids pickling it
    with every task, which was a major bottleneck with 10k+ voxels).
  - Switches from SLSQP + NonlinearConstraint to L-BFGS-B + bounds, removing
    the numerical constraint-Jacobian estimation that doubled function calls.

GPU path (requires CuPy): batch Adam optimizer — all voxels are optimized
simultaneously on GPU using a vectorized forward model:
    RF batch : (N, H*W) @ (H*W, T) → (N, T)  one GPU matmul per step
    Gradients: 4 finite-difference perturbations, all batched across N voxels
    Adam updates all N parameter vectors in parallel
  This mirrors the grid-fit GPU path in spirit: all the heavy work is a single
  GPU kernel, not N sequential Python calls.

Key functions:
    FinalFit_Vox()           — CPU: optimize one voxel (L-BFGS-B)
    get_final_estims()       — Dispatch to CPU or GPU path
    _get_final_estims_gpu()  — GPU: batch Adam optimizer for all voxels
"""

import numpy as np
import ctypes
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize

import popeye.utilities_cclab as utils

from H03_fit_utils import error_func
from H04_grid_predict import generate_grid_prediction
from H05_grid_fit import overload_estimate


# ---------------------------------------------------------------------------
# CPU path — Pool worker globals (set once via initializer, never pickled)
# ---------------------------------------------------------------------------

_stimulus = None
_bounds   = None


def _worker_init_finalfit(stimulus):
    """Set stimulus + bounds once per worker subprocess at Pool startup."""
    global _stimulus, _bounds
    _stimulus = stimulus
    max_deg   = float(stimulus.deg_x0.max())
    _bounds   = (
        (-max_deg * 2, max_deg * 2),   # x
        (-max_deg * 2, max_deg * 2),   # y
        (0.001, max_deg * 2),           # sigma
        (0.001, 2.0),                   # n (CSS exponent)
    )


# ---------------------------------------------------------------------------
# CPU path — per-voxel worker
# ---------------------------------------------------------------------------

def FinalFit_Vox(args):
    """
    Refine the pRF estimate for a single voxel/vertex via gradient descent.

    Uses L-BFGS-B with box bounds (faster than SLSQP + NonlinearConstraint —
    no constraint Jacobian estimation needed, ~2-4x fewer function calls).
    Stimulus is injected via Pool initializer — NOT pickled per task.

    Parameters
    ----------
    args : tuple
        (init_estim, unscaled_data)
        init_estim    : array-like (9,)  grid-fit estimate
        unscaled_data : ndarray (T,)     pre-unscaled voxel timeseries

    Returns
    -------
    tuple of 9 floats
        Best pRF estimate found (theta, r2, rho, sigma, n, x, y, beta, baseline).
    """
    init_estim, unscaled_data = args
    stimulus = _stimulus
    bounds   = _bounds

    if np.isnan(unscaled_data).any():
        return (np.nan,) * 9

    best_r2  = init_estim[1]
    best_fit = tuple(init_estim)
    x0       = [init_estim[5], init_estim[6], init_estim[3], init_estim[4]]

    try:
        result = minimize(
            error_func,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            args=(unscaled_data, stimulus, generate_grid_prediction),
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6},
        )
        overload_fin = overload_estimate(
            result.x, unscaled_data,
            generate_grid_prediction([*result.x, stimulus])
        )
        if overload_fin[1] > best_r2:
            best_fit = overload_fin
    except Exception:
        pass   # keep grid estimate if optimizer fails

    return best_fit


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def get_final_estims(gFit, param_width, timeseries_data, stimulus, fFit, indices,
                     use_gpu=False):
    """
    Run gradient-descent refinement for all voxels/vertices.

    CPU path (default):
        Parallel L-BFGS-B per voxel. Stimulus injected via Pool initializer so
        it is transmitted once per worker (not pickled with every voxel task).

    GPU path (use_gpu=True, requires CuPy):
        Batch Adam optimizer — all voxels optimized simultaneously on GPU.
        Forward model and finite-difference gradients are fully vectorized across
        all N voxels in each Adam step.

    Parameters
    ----------
    gFit : ndarray
        Grid-fit estimates array.
    param_width : list
        Search width for [x, y, sigma, n] (unused in current optimizer, kept
        for API compatibility).
    timeseries_data : ndarray
        Observed data (n_voxels, n_timepoints).
    stimulus : VisualStimulus
        Popeye stimulus object.
    fFit : ndarray
        Output array (shape is overridden; returned as (n_voxels, 9)).
    indices : list
        Indices into gFit for each voxel.
    use_gpu : bool
        If True, attempt CuPy GPU path.

    Returns
    -------
    fFit : ndarray, shape (n_voxels, 9)
    """
    nvoxs            = len(timeseries_data)
    timeseries_data  = np.asarray(timeseries_data, dtype=np.float64)

    if use_gpu:
        try:
            import cupy as cp
            print("GPU Final Fit: CuPy detected, using batch Adam optimizer.")
            return _get_final_estims_gpu(
                gFit, timeseries_data, stimulus, indices, nvoxs
            )
        except ImportError:
            print("CuPy not available — falling back to CPU path.")

    # ------------------------------------------------------------------
    # CPU path
    # ------------------------------------------------------------------
    fFit = np.empty((nvoxs, 9))

    # Pre-unscale data in main process (once, not in each worker)
    args_list = []
    for iin in range(nvoxs):
        idx = indices[iin]
        init_est = (gFit[idx[0], idx[1], idx[2], :] if isinstance(idx, (list, tuple))
                    else gFit[idx, :])
        y        = timeseries_data[iin]
        beta     = float(init_est[7])
        baseline = float(init_est[8])
        unscaled = (y - baseline) / (beta if abs(beta) > 1e-8 else 1e-8)
        args_list.append((np.asarray(init_est), unscaled))

    n_workers = cpu_count()
    chunksize = max(1, nvoxs // (n_workers * 4))

    with Pool(
        n_workers,
        initializer=_worker_init_finalfit,
        initargs=(stimulus,)
    ) as pool:
        results = list(tqdm(
            pool.imap(FinalFit_Vox, args_list, chunksize=chunksize),
            total=nvoxs, dynamic_ncols=True,
            desc="Final fit (CPU)"
        ))

    for i, result in enumerate(results):
        fFit[i, :] = result

    return fFit


# ---------------------------------------------------------------------------
# GPU path — batch Adam optimizer (requires CuPy)
# ---------------------------------------------------------------------------

def _get_final_estims_gpu(gFit, timeseries_data, stimulus, indices, nvoxs,
                           n_iter=300, lr=0.005, sub_batch=2000):
    """
    GPU-accelerated batch final fit using CuPy + Adam.

    All voxels in a sub-batch are optimized simultaneously:
      - Batch Gaussian RF:  (N, H*W) computed at once
      - Neural response:    (N, H*W) @ (H*W, T)  → (N, T)  one GPU matmul
      - HRF convolution:    batch FFT  (N, nfft)
      - Finite-diff grads:  4 extra forward passes, each handling all N voxels
      - Adam update:        (N, 4) parameter matrix updated in one step

    Parameters
    ----------
    n_iter : int
        Adam iterations per sub-batch (default 300).
    lr : float
        Adam learning rate (default 0.005).
    sub_batch : int
        Voxels per GPU sub-batch to fit within GPU memory (default 2000).
        Titan Xp 12GB: 2000 × ~10k-pixel RF uses ~160MB — well within budget.
    """
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as cp_fftconvolve

    max_deg = float(stimulus.deg_x0.max())
    nT      = stimulus.stim_arr.shape[2]

    # Transfer stimulus to GPU once
    deg_x_gpu    = cp.asarray(stimulus.deg_x,    dtype=cp.float32)        # (H, W)
    deg_y_gpu    = cp.asarray(stimulus.deg_y,    dtype=cp.float32)        # (H, W)
    stim_gpu     = cp.asarray(stimulus.stim_arr, dtype=cp.float32)        # (H, W, T)
    hrf_cpu      = utils.double_gamma_hrf(0, 1.3).astype(np.float32)
    hrf_gpu      = cp.asarray(hrf_cpu)

    stim_flat    = stim_gpu.reshape(-1, nT)                               # (H*W, T)
    deg_x_flat   = deg_x_gpu.ravel()                                      # (H*W,)
    deg_y_flat   = deg_y_gpu.ravel()                                      # (H*W,)
    dx           = float(cp.diff(deg_x_gpu[0, 0:2])[0])

    # Pre-compute HRF FFT for batch convolution
    nfft         = nT + len(hrf_cpu) - 1
    hrf_fft      = cp.fft.rfft(hrf_gpu, n=nfft)                          # (nfft//2+1,)

    # Bounds for parameter clipping
    p_lo = cp.array([-max_deg * 2, -max_deg * 2, 0.001, 0.001],          dtype=cp.float32)
    p_hi = cp.array([ max_deg * 2,  max_deg * 2, max_deg * 2, 2.0],      dtype=cp.float32)

    # ------------------------------------------------------------------
    def _forward_batch(params):
        """
        Batch CSS pRF forward model.

        Parameters
        ----------
        params : cupy ndarray, shape (N, 4)  — [x, y, sigma, n]

        Returns
        -------
        pred : cupy ndarray, shape (N, T)
        """
        x     = params[:, 0]   # (N,)
        y     = params[:, 1]
        sigma = params[:, 2]
        n     = params[:, 3]

        # Batch 2D Gaussian RF: (N, H*W)
        dx_diff = deg_x_flat[None, :] - x[:, None]    # (N, H*W)
        dy_diff = deg_y_flat[None, :] - y[:, None]
        rf      = cp.exp(-(dx_diff**2 + dy_diff**2) / (2.0 * sigma[:, None]**2))
        rf     /= (2.0 * np.pi * sigma[:, None]**2) / (dx**2)

        # Neural response: (N, H*W) @ (H*W, T) → (N, T)
        response = rf @ stim_flat

        # CSS compressive nonlinearity
        response = cp.abs(response) ** n[:, None]

        # Batch HRF convolution via FFT
        R_fft = cp.fft.rfft(response, n=nfft)                            # (N, nfft//2+1)
        pred  = cp.fft.irfft(R_fft * hrf_fft[None, :])[:, :nT]          # (N, T)

        # Normalize to percent signal change
        mu   = pred.mean(axis=1, keepdims=True)
        pred = (pred - mu) / (cp.abs(mu) + 1e-8)
        return pred

    def _sse_batch(params, data):
        """SSE for each voxel: (N,)."""
        pred = _forward_batch(params)
        return cp.sum((data - pred)**2, axis=1)

    def _grad_batch(params, data, eps=1e-4):
        """Finite-difference gradients, vectorized across all N voxels: (N, 4)."""
        loss0 = _sse_batch(params, data)                                  # (N,)
        grads = cp.empty_like(params)
        for p in range(4):
            p_plus      = params.copy()
            p_plus[:, p] += eps
            grads[:, p]  = (_sse_batch(p_plus, data) - loss0) / eps
        return grads
    # ------------------------------------------------------------------

    fFit_out = np.empty((nvoxs, 9))

    print(f"GPU Final Fit: {nvoxs} voxels | {n_iter} Adam steps | sub_batch={sub_batch}")
    pbar = tqdm(total=nvoxs, desc="Final fit (GPU)", dynamic_ncols=True)

    for start in range(0, nvoxs, sub_batch):
        end = min(start + sub_batch, nvoxs)
        B   = end - start

        # Collect initial estimates + unscaled data for this sub-batch
        params_np  = np.zeros((B, 4), dtype=np.float32)
        data_np    = np.zeros((B, nT), dtype=np.float32)
        init_ests  = []

        for i in range(B):
            iin      = start + i
            idx      = indices[iin]
            init_est = (gFit[idx[0], idx[1], idx[2], :]
                        if isinstance(idx, (list, tuple)) else gFit[idx, :])
            init_ests.append(np.asarray(init_est, dtype=np.float64))
            params_np[i]  = [init_est[5], init_est[6], init_est[3], init_est[4]]
            y             = timeseries_data[iin].astype(np.float32)
            beta          = float(init_est[7])
            baseline      = float(init_est[8])
            data_np[i]    = (y - baseline) / (beta if abs(beta) > 1e-8 else 1e-8)

        params_gpu = cp.asarray(params_np)
        data_gpu   = cp.asarray(data_np)

        # Adam optimizer — all B voxels updated in parallel each step
        m      = cp.zeros_like(params_gpu)
        v      = cp.zeros_like(params_gpu)
        b1, b2 = 0.9, 0.999
        eps_a  = 1e-8

        for t in range(1, n_iter + 1):
            grads    = _grad_batch(params_gpu, data_gpu)
            m        = b1 * m + (1 - b1) * grads
            v        = b2 * v + (1 - b2) * grads**2
            m_hat    = m / (1 - b1**t)
            v_hat    = v / (1 - b2**t)
            params_gpu -= lr * m_hat / (cp.sqrt(v_hat) + eps_a)
            params_gpu  = cp.clip(params_gpu, p_lo, p_hi)

        # Extract final predictions and compute overload estimates on CPU
        params_final = cp.asnumpy(params_gpu)
        preds_final  = cp.asnumpy(_forward_batch(params_gpu))

        for i in range(B):
            iin      = start + i
            init_est = init_ests[i]
            try:
                est = overload_estimate(params_final[i], data_np[i], preds_final[i])
                fFit_out[iin, :] = est if est[1] >= init_est[1] else init_est
            except Exception:
                fFit_out[iin, :] = init_est

        pbar.update(B)

    pbar.close()

    # Free GPU memory
    del deg_x_gpu, deg_y_gpu, stim_gpu, hrf_gpu, stim_flat, deg_x_flat
    del deg_y_flat, hrf_fft
    cp.get_default_memory_pool().free_all_blocks()

    return fFit_out

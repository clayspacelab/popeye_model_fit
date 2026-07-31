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
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize


from H03_fit_utils import error_func
from H04_grid_predict import generate_grid_prediction
from H05_grid_fit import overload_estimate


# ---------------------------------------------------------------------------
# CPU path — Pool worker globals (set once via initializer, never pickled)
# ---------------------------------------------------------------------------

_stimulus = None
_bounds   = None
_hrf      = None


def _worker_init_finalfit(stimulus, hrf):
    """Set stimulus + bounds once per worker subprocess at Pool startup. Also set HRF."""
    global _stimulus, _bounds, _hrf
    _stimulus = stimulus
    _hrf = hrf
    max_deg   = float(stimulus.deg_x.max())
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
    hrf      = _hrf

    if np.isnan(unscaled_data).any():
        return (np.nan,) * 9

    best_r2  = init_estim[1]
    best_fit = tuple(init_estim)
    x0       = [init_estim[5], init_estim[6], init_estim[3], init_estim[4]]

    def pred_func(args):
        args = (*args, stimulus, hrf)
        return generate_grid_prediction(args)
    
    try:
        result = minimize(
            error_func,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            args=(unscaled_data, pred_func),
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6},
        )
        overload_fin = overload_estimate(
            result.x, unscaled_data,
            pred_func(result.x)
        )
        if overload_fin[1] > best_r2:
            best_fit = overload_fin
    except Exception as e:
        print(e)
        #pass   # keep grid estimate if optimizer fails

    return best_fit


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def get_final_estims(gFit, param_width, timeseries_data, stimulus, hrf, fFit, indices,
                     use_gpu=False, n_iter=300, lr=0.005, sub_batch=None):
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
    hrf : ndarray
        Hemodynamic response function.
    fFit : ndarray
        Output array (shape is overridden; returned as (n_voxels, 9)).
    indices : list
        Indices into gFit for each voxel.
    use_gpu : bool
        If True, attempt CuPy GPU path.
    n_iter : int
        Adam iterations per sub-batch (GPU path only; default 300).
    lr : float
        Adam learning rate (GPU path only; default 0.005).
    sub_batch : int or None
        Voxels per GPU sub-batch. None (default) sizes it automatically from
        free VRAM. GPU path only.

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
                gFit, timeseries_data, stimulus, hrf, indices, nvoxs,
                n_iter=n_iter, lr=lr, sub_batch=sub_batch
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
        initargs=(stimulus, hrf)
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

def _get_final_estims_gpu(gFit, timeseries_data, stimulus, hrf, indices, nvoxs,
                           n_iter=300, lr=0.005, sub_batch=None):
    """
    GPU-accelerated batch final fit using CuPy + Adam.

    All voxels in a sub-batch are optimized simultaneously:
      - Batch Gaussian RF:  (N, H*W) computed at once
      - Neural response:    (N, H*W) @ (H*W, T)  → (N, T)  one GPU matmul
      - HRF convolution:    batch FFT  (N, nfft), nfft rounded up to a fast length
      - Finite-diff grads:  4 perturbations. The n-perturbation reuses the cached
        linear neural response (RF + matmul unchanged when only the exponent
        moves), so it costs only the cheap nonlinearity + HRF conv.
      - Adam update:        (N, 4) parameter matrix updated in one step

    Parameters
    ----------
    n_iter : int
        Adam iterations per sub-batch (default 300).
    lr : float
        Adam learning rate (default 0.005).
    sub_batch : int or None
        Voxels per GPU sub-batch. None (default) sizes it automatically from
        free VRAM (larger batches keep the GPU saturated and cut Python/kernel
        launch overhead — the dominant cost at the old fixed 2000).
    """
    import cupy as cp

    max_deg = float(stimulus.deg_x.max())
    nT      = stimulus.run_length #stimulus.stim_arr.shape[2]

    # Transfer stimulus to GPU once
    deg_x_gpu    = cp.asarray(stimulus.deg_x,    dtype=cp.float32)        # (H, W)
    deg_y_gpu    = cp.asarray(stimulus.deg_y,    dtype=cp.float32)        # (H, W)
    #stim_gpu     = cp.asarray(stimulus.stim_arr, dtype=cp.float32)        # (H, W, T)
    hrf_cpu      = hrf.astype(np.float32) #should already be float32, but just in case
    hrf_gpu      = cp.asarray(hrf_cpu)

    stim_flat    = cp.asarray(stimulus.stim_arr.T,
                            dtype=cp.float32)                             # (H*W, T)
    #stim_gpu.reshape(-1, nT)   
    deg_x_flat   = deg_x_gpu.ravel()                                      # (H*W,)
    deg_y_flat   = deg_y_gpu.ravel()                                      # (H*W,)
    dx           = float(cp.diff(deg_x_gpu[0, 0:2])[0])
    HW           = deg_x_flat.size

    # Pre-compute HRF FFT for batch convolution. Rounding nfft UP to a
    # 5-smooth "fast" length only zero-pads the transform, so the linear
    # convolution result in [:nT] is bit-for-bit the same — but cuFFT runs
    # much faster on fast lengths than on the raw (nT + len(hrf) - 1).
    nfft         = nT + len(hrf_cpu) - 1
    try:
        from cupyx.scipy.fft import next_fast_len
        nfft = int(next_fast_len(nfft))
    except Exception:
        pass
    hrf_fft      = cp.fft.rfft(hrf_gpu, n=nfft)                          # (nfft//2+1,)

    # Auto-size the sub-batch from free VRAM when not overridden. Peak memory is
    # dominated by a few (B, H*W) float32 temporaries in the forward pass; use a
    # generous per-voxel estimate and a 60% headroom cap for safety.
    if sub_batch is None:
        free_bytes, _ = cp.cuda.Device().mem_info
        bytes_per_vox = HW * 4 * 6          # ~3 (B,H*W) temporaries + fft + margin
        sub_batch = int((free_bytes * 0.6) / max(bytes_per_vox, 1))
        sub_batch = int(np.clip(sub_batch, 500, nvoxs))

    # Bounds for parameter clipping
    p_lo = cp.array([-max_deg * 2, -max_deg * 2, 0.001, 0.001],          dtype=cp.float32)
    p_hi = cp.array([ max_deg * 2,  max_deg * 2, max_deg * 2, 2.0],      dtype=cp.float32)

    # ------------------------------------------------------------------
    def _neural_response(params):
        """Linear (pre-nonlinearity) neural response: depends on x, y, sigma only.

        Returns (N, T). Split out from the forward model so the exponent-gradient
        perturbation can reuse it without rebuilding the RF or redoing the matmul.
        """
        x     = params[:, 0]   # (N,)
        y     = params[:, 1]
        sigma = params[:, 2]

        # Batch 2D Gaussian RF: (N, H*W)
        dx_diff = deg_x_flat[None, :] - x[:, None]    # (N, H*W)
        dy_diff = deg_y_flat[None, :] - y[:, None]
        rf      = cp.exp(-(dx_diff**2 + dy_diff**2) / (2.0 * sigma[:, None]**2))
        #rf     /= (2.0 * np.pi * sigma[:, None]**2) / (dx**2)

        # Neural response: (N, H*W) @ (H*W, T) → (N, T)
        return rf @ stim_flat

    def _forward_from_response(response, n):
        """Apply CSS nonlinearity + HRF conv + PSC normalize to a linear response.

        response : (N, T) linear neural response
        n        : (N,)   CSS exponent
        """
        response = response ** n[:, None]

        # Batch HRF convolution via FFT
        R_fft = cp.fft.rfft(response, n=nfft)                            # (N, nfft//2+1)
        pred  = cp.fft.irfft(R_fft * hrf_fft[None, :])[:, :nT]          # (N, T)

        # Normalize to percent signal change
        # mu   = pred.mean(axis=1, keepdims=True)
        # pred = (pred - mu) / (cp.abs(mu) + 1e-8)
        return pred

    def _forward_batch(params):
        """Batch CSS pRF forward model. params (N, 4) [x, y, sigma, n] → pred (N, T)."""
        return _forward_from_response(_neural_response(params), params[:, 3])

    def _grad_batch(params, data, eps=1e-4):
        """Finite-difference gradients, vectorized across all N voxels: (N, 4).

        The x/y/sigma perturbations each need a full forward pass (the RF and
        matmul change). The n perturbation reuses the cached linear response —
        only the nonlinearity + HRF conv are recomputed — saving one RF build
        and one (N, H*W)@(H*W, T) matmul per gradient evaluation.
        """
        resp0 = _neural_response(params)                                  # (N, T)
        pred0 = _forward_from_response(resp0, params[:, 3])
        loss0 = cp.sum((data - pred0)**2, axis=1)                         # (N,)

        grads = cp.empty_like(params)
        for p in range(3):   # x, y, sigma — RF/matmul change, full recompute
            p_plus       = params.copy()
            p_plus[:, p] += eps
            pred_p        = _forward_from_response(_neural_response(p_plus),
                                                   p_plus[:, 3])
            grads[:, p]   = (cp.sum((data - pred_p)**2, axis=1) - loss0) / eps

        # n (exponent) — reuse resp0; only nonlinearity + conv recomputed
        pred_n      = _forward_from_response(resp0, params[:, 3] + eps)
        grads[:, 3] = (cp.sum((data - pred_n)**2, axis=1) - loss0) / eps
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
    del deg_x_gpu, deg_y_gpu, hrf_gpu, stim_flat, deg_x_flat
    del deg_y_flat, hrf_fft
    cp.get_default_memory_pool().free_all_blocks()

    return fFit_out

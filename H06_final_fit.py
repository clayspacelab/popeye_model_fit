"""
H06_final_fit.py — Gradient-descent refinement of pRF estimates.

CPU/GPU path: independent JAXopt fits in vmapped chunks.

Key functions:
    _final_fit_minimize_jit — JAX loss function used by JAXopt
    get_final_estims()       — main optimizer function
"""

import numpy as np
import jax
import os
#jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_debug_infs', True)
#jax.config.update("jax_disable_jit", True)
#jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jaxopt import ProjectedGradient
from jaxopt import projection
#from tqdm import tqdm


from H03_fit_utils import error_func_min
from H04_grid_predict import generate_grid_prediction_jax
from H05_grid_fit import overload_estimate_jax



def _final_fit_minimize_jax(parameters, data, deg_x, deg_y, stim_arr, hrf):
    """JAX residual matching error_func_lsq's original computation."""
    return error_func_min(
        parameters, data,
        lambda current_parameters: generate_grid_prediction_jax(
            current_parameters, deg_x, deg_y, stim_arr, hrf
        ),
    )

def _final_fit_predictions_batch_jax(parameters, deg_x, deg_y, stim_arr, hrf):
    return jax.vmap(
        generate_grid_prediction_jax,
        in_axes=(0, None, None, None, None),
    )(parameters, deg_x, deg_y, stim_arr, hrf)


_final_fit_predictions_batch_jit = jax.jit(
    _final_fit_predictions_batch_jax
)

_final_fit_overloads_batch_jit = jax.jit(
    jax.vmap(overload_estimate_jax, in_axes=(0, 0, 0))
)


def _estimate_jax_sub_batch(stimulus, n_timepoints, n_voxels):
    """Estimate a conservative JAX/Optimistix batch size from available VRAM."""
    try:
        #THIS SHOULD BE REPLACED BY A CPU/GPU CHECK
        #cause you could use jax.devices('gpu')[0] if a gpu is available
        device = next(
            device for device in jax.devices()
            if device.platform == "gpu"
        )
        ### possible alternative
        # gpu_devices = [
        #     device for device in jax.devices()
        #     if device.platform == "gpu"
        # ]
        # if not gpu_devices:
        #     return min(16, n_voxels)
        # device = gpu_devices[0]

        memory_stats = device.memory_stats()
        if memory_stats is None:
            raise RuntimeError("JAX did not provide GPU memory statistics")

        total_bytes = int(memory_stats["bytes_limit"])
        used_bytes = int(memory_stats.get("bytes_in_use", 0))
        free_bytes = max(0, total_bytes - used_bytes)
        # RF construction, residuals, autodiff/Jacobian workspace, and
        # Optimistix state can coexist during a solve. Reserve most free VRAM.
        pixels = int(np.asarray(stimulus.deg_x).size)
        bytes_per_voxel = 8 * (12 * pixels + 32 * n_timepoints + 256)
        usable_bytes = int(free_bytes * 0.05)
        estimated = usable_bytes // max(bytes_per_voxel, 1)
        batch_size = int(np.clip(estimated, 1, n_voxels))
        print(
            f"JAX VRAM sizing: free={free_bytes / 2**30:.2f} GiB / "
            f"total={total_bytes / 2**30:.2f} GiB | "
            f"estimated chunk={batch_size}"
        )
        return batch_size
    except (StopIteration, KeyError, TypeError, RuntimeError, ValueError):
        # CPU JAX or unavailable GPU memory reporting: keep the batch small.
        print('CPU ONLY!')
        return min(16, n_voxels)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def get_final_estims(gFit, timeseries_data, stimulus, hrf, fFit, indices,
                     use_gpu=False, n_iter=300, lr=0.005, sub_batch=None):
    """
    Run projection-gradient refinement for all voxels/vertices.


    Parameters
    ----------
    gFit : ndarray
        Grid-fit estimates array.
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
        Voxels per JAX chunk. None estimates a conservative size
        from available VRAM. [NOT FULLY IMPLEMENTED]

    Returns
    -------
    fFit : ndarray, shape (n_voxels, 9)
    """
    nvoxs            = len(timeseries_data)
    timeseries_data  = np.asarray(timeseries_data, dtype=np.float32)

    args_list = []
    for iin in range(nvoxs):
        idx = indices[iin]
        init_est = (gFit[idx[0], idx[1], idx[2], :] if isinstance(idx, (list, tuple))
                    else gFit[idx, :])
        y        = timeseries_data[iin]
        args_list.append((np.asarray(init_est), y))

    deg_x = jnp.asarray(stimulus.deg_x, dtype=float)
    deg_y = jnp.asarray(stimulus.deg_y, dtype=float)
    stim_arr = jnp.asarray(stimulus.stim_arr, dtype=float)
    hrf_jax = jnp.asarray(hrf, dtype=float)
    max_deg = float(stimulus.deg_x.max())
    bounds = (
        [-max_deg * 2, -max_deg * 2, 0.1, 0.01],
        [max_deg * 2, max_deg * 2, max_deg * 2, 2.0],
    )
    lower = jnp.asarray(bounds[0], dtype=float)
    upper = jnp.asarray(bounds[1], dtype=float)


    initial_estimates = np.asarray([init_est for init_est, _ in args_list])
    initial_params = np.asarray([
        [init_est[5], init_est[6], init_est[3], init_est[4]]
        for init_est, _ in args_list
    ]) #, dtype=np.float32)


    ###Create projection function for enforcing contstraints
    #define constraints
    constraints = {
        'sigma_lower' : lower[2],
        'sigma_upper' : upper[2],
        'n_lower' : lower[3],
        'n_upper' : upper[3],
        'xy_radius_base' : max_deg,
    }

    def composite_projection(params, hp):
        sigma = projection.projection_box(
            params[2:3],
            (hp["sigma_lower"],
            hp["sigma_upper"]),
        )

        n = projection.projection_box(
            params[3:4],
            (hp["n_lower"],
            hp["n_upper"]),
        )

        radius = jnp.minimum(
            2.0 * hp["xy_radius_base"],
            hp["xy_radius_base"] + 2.0 * sigma[0],
        )

        xy = projection.projection_l2_ball(
            params[0:2],
            max_value=radius,
        )

        return jnp.hstack((xy,sigma,n))

    finite_mask = np.isfinite(timeseries_data).all(axis=1)
    fit_data = np.where(finite_mask[:, None], timeseries_data, 0.0)

    if sub_batch is None:
        sub_batch = _estimate_jax_sub_batch(
            stimulus, timeseries_data.shape[1], nvoxs
        )
    sub_batch = 50000 #max(1, int(sub_batch))
    print(f"Final fit (JAX/Optimistix): {nvoxs} voxels | chunk={sub_batch}")

    solver = ProjectedGradient(
        fun=_final_fit_minimize_jax,
        projection=composite_projection,
        maxiter=500,
        tol=1e-3,
        stepsize=1e-3,
        acceleration=True,
        verbose=False
    )

    def fit_one_voxel(init_params, data, deg_x, deg_y, stim_arr, hrf):
        result = solver.run(
            init_params,
            constraints,
            data,
            deg_x,
            deg_y,
            stim_arr,
            hrf,
        )
        return result


    @jax.jit(static_argnames='batch_size')
    def fit_many_voxels(batched_params,static_params,batch_size):

        def fit_one(params):
            return fit_one_voxel(*params,*static_params)
        
        return jax.lax.map(
            fit_one,batched_params,batch_size=batch_size)


    batch_data = jnp.asarray(fit_data, dtype=float)
    batch_params = initial_params

    result = fit_many_voxels((batch_params, batch_data), 
                              (deg_x, deg_y, stim_arr, hrf),
                                batch_size=sub_batch)

    params_final = result.params

    predictions = _final_fit_predictions_batch_jit(
        params_final, deg_x, deg_y, stim_arr, hrf_jax
    )
    estimates = np.asarray(_final_fit_overloads_batch_jit(
        params_final, batch_data, predictions
    ))
    selected = np.where(
        estimates[:, 1:2] > initial_estimates[:, 1:2],
        estimates,
        initial_estimates,
    )
    selected[~finite_mask] = np.nan
    fFit = selected   

    return fFit
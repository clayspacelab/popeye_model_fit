"""
H06_final_fit.py — Gradient-descent refinement of pRF estimates.

CPU/GPU path: independent JAXopt fits in vmapped chunks.

Key functions:
    _final_fit_minimize_jit — JAX loss function used by JAXopt
    get_final_estims()       — main optimizer function
"""

import numpy as np
import jax
#jax.config.update("jax_enable_x64", True)
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_debug_infs', True)
#jax.config.update("jax_disable_jit", True)
#jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jaxopt import ProjectedGradient
from jaxopt import projection

from .fit_utils import error_func_min, results_to_img
from .grid_predict import generate_grid_prediction_jax
from .grid_fit import overload_estimate_jax



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

#WOULD LIKE TO IMPLEMENT THIS EVENTUALLY BUT CURRENT VERSION NOT ACCURATE
# def _estimate_jax_sub_batch(stimulus, n_timepoints, n_voxels):
#     """Estimate a conservative JAX/JAXOpt batch size from available VRAM."""
#     try:
#         #THIS SHOULD BE REPLACED BY A CPU/GPU CHECK
#         #cause you could use jax.devices('gpu')[0] if a gpu is available
#         device = next(
#             device for device in jax.devices()
#             if device.platform == "gpu"
#         )
#         ### possible alternative
#         # gpu_devices = [
#         #     device for device in jax.devices()
#         #     if device.platform == "gpu"
#         # ]
#         # if not gpu_devices:
#         #     return min(16, n_voxels)
#         # device = gpu_devices[0]

#         memory_stats = device.memory_stats()
#         if memory_stats is None:
#             raise RuntimeError("JAX did not provide GPU memory statistics")

#         total_bytes = int(memory_stats["bytes_limit"])
#         used_bytes = int(memory_stats.get("bytes_in_use", 0))
#         free_bytes = max(0, total_bytes - used_bytes)
#         # RF construction, residuals, autodiff/Jacobian workspace, and
#         # JAXOpt state can coexist during a solve. Reserve most free VRAM.
#         pixels = int(np.asarray(stimulus.deg_x).size)
#         bytes_per_voxel = 8 * (12 * pixels + 32 * n_timepoints + 256)
#         usable_bytes = int(free_bytes * 0.05)
#         estimated = usable_bytes // max(bytes_per_voxel, 1)
#         batch_size = int(np.clip(estimated, 1, n_voxels))
#         print(
#             f"JAX VRAM sizing: free={free_bytes / 2**30:.2f} GiB / "
#             f"total={total_bytes / 2**30:.2f} GiB | "
#             f"estimated chunk={batch_size}"
#         )
#         return batch_size
#     except (StopIteration, KeyError, TypeError, RuntimeError, ValueError):
#         # CPU JAX or unavailable GPU memory reporting: keep the batch small.
#         print('CPU ONLY!')
#         return min(16, n_voxels)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def get_final_estims(gFit, timeseries_data, stimulus, hrf, fFit, indices,
                     batch_size=50000):
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
    batch_size : int
        Voxels per JAX chunk.

    Returns
    -------
    fFit : ndarray, shape (n_voxels, 9)
    """


    #### prepare data

    #I'm not sure this check should be here...should have been handled by now, but leaving for now
    finite_mask = np.isfinite(timeseries_data).all(axis=1)
    fit_data = np.where(finite_mask[:, None], timeseries_data, 0.0)


    #### prepare parameters

    idx = np.asarray(indices)
    if gFit.ndim == 4:
        # Indices should be (x, y, z) voxel coordinates.
        initial_estimates = gFit[
            idx[:, 0],
            idx[:, 1],
            idx[:, 2],
            :,
        ]
    elif gFit.ndim == 2:
        # Indices should be direct indices into the first axis of gFit surface.
        initial_estimates = gFit[idx, :]
    else:
        raise ValueError(
                f"Expected gFit to have 2 dimensions (surface) or 4 dimensions "
                f"(3D NIfTI), got shape {gFit.shape}."
        )
    initial_params = initial_estimates[:, [5, 6, 3, 4]].copy()


    ### Create projection function for enforcing contstraints
    #NOTE: eventually this should be made configurable or at least more modular.

    max_deg = stimulus.deg_x.max()

    #define actual constraints
    constraints = {
        'sigma_lower' : 0.1, #ounds['lower'][2],
        'sigma_upper' : max_deg * 2, #bounds['upper'][2],
        'n_lower' : 0.01, #bounds['lower'][3],
        'n_upper' : 2.0, #bounds['upper'][3],
        'xy_radius_base' : max_deg,
    }

    #composite projection function for enforcing constraints on all parameters simultaneously
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

    ### construct solver for JAXopt projected gradient descent

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


    # if sub_batch is None:
    #     sub_batch = _estimate_jax_sub_batch(
    #         stimulus, timeseries_data.shape[1], nvoxs
    #     )
    # sub_batch = 50000 #max(1, int(sub_batch))

    @jax.jit(static_argnames='bsize')
    def fit_many_voxels(batched_params,static_params,bsize):

        def fit_one(params):
            return fit_one_voxel(*params,*static_params)
        
        return jax.lax.map(
            fit_one,batched_params,batch_size=bsize)


    #### fit and process results

    print(f"Final fit (JAXOpt): {fit_data.shape[0]} voxels | batch size={batch_size}")

    #prepare data/parameters for JAX (not strictly necessary)
    batch_data = jnp.asarray(fit_data, dtype=float)
    batch_params = jnp.asarray(initial_params, dtype=float)
    deg_x = jnp.asarray(stimulus.deg_x, dtype=float)
    deg_y = jnp.asarray(stimulus.deg_y, dtype=float)
    stim_arr = jnp.asarray(stimulus.stim_arr, dtype=float)
    hrf_jax = jnp.asarray(hrf, dtype=float)

    result = fit_many_voxels((batch_params, batch_data), 
                              (deg_x, deg_y, stim_arr, hrf_jax),
                                bsize=batch_size)

    params_final = result.params

    predictions = _final_fit_predictions_batch_jit(
        params_final, deg_x, deg_y, stim_arr, hrf_jax
    )
    estimates = jax.device_get(_final_fit_overloads_batch_jit(
        params_final, batch_data, predictions
    ))

    #NOTE: we currently are NOT preventing negative betas in final fit.
    selected = np.where(
        estimates[:, 1:2] > initial_estimates[:, 1:2],
        estimates,
        initial_estimates,
    )
    selected[~finite_mask] = np.nan

    fFit = results_to_img(fFit,selected,indices)

    return fFit
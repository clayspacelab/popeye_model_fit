"""
H06_final_fit.py — Gradient-descent refinement of pRF estimates.

Takes grid-fit estimates as initial values and refines them using
scipy.optimize.minimize (SLSQP) with eccentricity constraints.

Key functions:
    FinalFit_Vox()       — Optimize one voxel/vertex
    get_final_estims()   — Parallel final fitting across all voxels/vertices
"""

import numpy as np
import ctypes
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize, NonlinearConstraint

import popeye.utilities_cclab as utils

from H03_fit_utils import error_func
from H04_grid_predict import generate_grid_prediction
from H05_grid_fit import overload_estimate


def FinalFit_Vox(args):
    """
    Refine the pRF estimate for a single voxel/vertex via gradient descent.

    Uses SLSQP optimization with bounds and nonlinear eccentricity constraints.
    The grid estimate is used directly as the initial guess.

    Parameters
    ----------
    args : tuple
        (init_estim, param_width, timeseries_data, stimulus, use_gpu)

        init_estim : array-like (9,)
            Grid-fit estimate (theta, r2, rho, sigma, n, x, y, beta, baseline).
        param_width : list
            Search width for [x, y, sigma, n] (used for bounds generation).
        timeseries_data : ndarray
            Observed BOLD timeseries for this voxel/vertex.
        stimulus : VisualStimulus
            Popeye stimulus object.
        use_gpu : bool
            GPU flag (reserved for future use).

    Returns
    -------
    best_fit : tuple of 9 floats
        Best pRF estimate found (theta, r2, rho, sigma, n, x, y, beta, baseline).
    """
    init_estim, param_width, timeseries_data, stimulus, use_gpu = args

    x_estim = init_estim[5]
    y_estim = init_estim[6]
    sigma_estim = init_estim[3]
    n_estim = init_estim[4]
    beta_estim = init_estim[7]
    baseline_estim = init_estim[8]

    # Check for NaN data
    if np.isnan(timeseries_data).any():
        return (np.nan,) * 9

    # Unscale data using grid-fit beta and baseline
    unscaled_data = (timeseries_data - baseline_estim) / beta_estim

    # Bounds: wide search range based on stimulus extent
    max_deg = stimulus.deg_x0.max()
    bounds = (
        (-max_deg * 2, max_deg * 2),      # x
        (-max_deg * 2, max_deg * 2),      # y
        (0.001, max_deg * 2),             # sigma
        (0.001, 2),                        # n (CSS exponent)
    )

    # Nonlinear constraints to keep RF within plausible eccentricity
    constraints = (
        NonlinearConstraint(
            lambda x: np.sqrt(x[0]**2 + x[1]**2),
            -np.inf, 2 * max_deg
        ),
        NonlinearConstraint(
            lambda x: np.sqrt(x[0]**2 + x[1]**2) - 2 * x[2],
            -np.inf, max_deg
        ),
    )

    # Initialize with grid estimate
    best_r2 = init_estim[1]
    best_fit = init_estim

    # Use grid estimate directly as initial guess
    x_guess, y_guess = x_estim, y_estim
    sigma_guess, n_guess = sigma_estim, n_estim

    try:
        finfit = minimize(
            error_func,
            [x_guess, y_guess, sigma_guess, n_guess],
            bounds=bounds,
            method='SLSQP',
            args=(unscaled_data, stimulus, generate_grid_prediction),
            constraints=constraints,
        )
        overload_finestim = overload_estimate(
            finfit.x, unscaled_data,
            generate_grid_prediction([*finfit.x, stimulus])
        )
        if overload_finestim[1] > best_r2:
            best_r2 = overload_finestim[1]
            best_fit = overload_finestim
    except ValueError:
        pass  # Keep grid estimate if optimization fails

    return best_fit


def get_final_estims(gFit, param_width, timeseries_data, stimulus, fFit, indices,
                     use_gpu=False):
    """
    Run gradient-descent refinement for all voxels/vertices in parallel.

    Parameters
    ----------
    gFit : ndarray
        Grid-fit estimates array.
    param_width : list
        Search width for [x, y, sigma, n].
    timeseries_data : ndarray
        Observed data (n_voxels, n_timepoints).
    stimulus : VisualStimulus
        Popeye stimulus object.
    fFit : ndarray
        Output array to fill with final fit results.
    indices : list
        Indices into gFit/fFit for each voxel/vertex.
        - Volumetric: list of (x, y, z) tuples
        - Surface: list of int
    use_gpu : bool
        GPU flag (reserved for future use in final fit).

    Returns
    -------
    fFit : ndarray
        Updated final fit array.
    """
    timeseries_data = utils.generate_shared_array(timeseries_data, ctypes.c_double)
    gFit = utils.generate_shared_array(gFit, ctypes.c_double)

    fFit = np.empty((timeseries_data.shape[0], 9))
    nvoxs = len(timeseries_data)

    # Build args depending on index format
    args_list = []
    for iin in range(nvoxs):
        idx = indices[iin]
        if isinstance(idx, (list, tuple)):
            init_est = gFit[idx[0], idx[1], idx[2], :]  # volumetric
        else:
            init_est = gFit[idx, :]  # surface
        args_list.append(
            (init_est, param_width, timeseries_data[iin, :], stimulus, use_gpu)
        )

    with Pool(cpu_count()) as pool:
        results = []
        for result in tqdm(pool.imap(FinalFit_Vox, args_list),
                           total=nvoxs, dynamic_ncols=False):
            results.append(result)

    for i, result in enumerate(results):
        idx = indices[i]
        if isinstance(idx, (list, tuple)):
            fFit[idx[0], idx[1], idx[2], :] = result  # volumetric
        else:
            fFit[idx, :] = result  # surface

    return fFit

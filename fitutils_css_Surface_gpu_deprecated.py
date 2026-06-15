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

# from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries_nomask
import popeye.utilities_cclab as utils

from fit_utils import *
import jax
import jax.numpy as jnp
from jaxopt import LBFGSB
from jax.scipy.special import gamma

def generate_og_receptive_field_jax(x, y, sigma, deg_x, deg_y):
    """
    Generate a 2D Gaussian receptive field in JAX.
    deg_x, deg_y: 2D arrays (meshgrid)
    x, y, sigma: scalars
    """
    d = (deg_x - x)**2 + (deg_y - y)**2
    rf = jnp.exp(-d / (2.0 * sigma**2))
    return rf

def generate_rf_timeseries_nomask_jax(stim_arr, rf):
    """
    stim_arr: shape (xlim, ylim, zlim)  # zlim = time
    rf: shape (xlim, ylim)
    Returns: 1D array (zlim,)
    """
    # Flatten rf to 1D
    rf_flat = rf.ravel()
    # Reshape stim_arr to (xlim*ylim, zlim)
    stim_flat = stim_arr.reshape(-1, stim_arr.shape[2])
    # Dot product: for each timepoint, sum stim_flat[:,k] * rf_flat
    # This is (rf_flat @ stim_flat) = (zlim,)
    return jnp.dot(rf_flat, stim_flat)

def double_gamma_hrf_jax(delay, tr, fptr=1.0, integrator=None):
    """
    JAX-compatible double gamma hemodynamic response function (HRF).

    Parameters
    ----------
    delay : float
        The delay of the HRF peak and undershoot.
    tr : float
        The length of the repetition time in seconds.
    fptr : float
        The number of stimulus frames per repetition time (unused here).
    integrator : callable or None
        Integration function for normalization (default: None).

    Returns
    -------
    hrf : jnp.ndarray
        The hemodynamic response function.
    """
    alpha_1 = 5.0 + delay
    beta_1 = 1.0
    c = 0.1
    alpha_2 = 15.0 + delay
    beta_2 = 1.0

    t = jnp.arange(0, 32, tr)

    hrf = (((t ** alpha_1) * (beta_1 ** alpha_1) * jnp.exp(-beta_1 * t)) / gamma(alpha_1)) - \
          c * (((t ** alpha_2) * (beta_2 ** alpha_2) * jnp.exp(-beta_2 * t)) / gamma(alpha_2))

    # Normalize if integrator is provided (default: area under curve)
    if integrator is not None:
        hrf = hrf / integrator(hrf, t)
    else:
        # Default normalization with trapezoidal integration
        hrf = hrf / jnp.trapz(hrf, t)

    return hrf

def generate_grid_prediction_jax(params, stimulus):
    # stimulus, x, y, sigma, n = args
    x, y, sigma, n = params
    # Generate RF
    rf = generate_og_receptive_field_jax(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
    # rf = utils.generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
    rf /= ((2 * jnp.pi * sigma**2) * 1/jnp.diff(stimulus.deg_x[0,0:2])**2)

    # Extract the stimulus time-series
    response = generate_rf_timeseries_nomask_jax(stimulus.stim_arr, rf)
    # response = utils.generate_rf_timeseries(stimulus.stim_arr, rf)
    response = response ** n
    predsig = jnp.convolve(response, double_gamma_hrf_jax(0, 1.3), mode='full')[0:len(response)]
    # Normalize the units
    predsig = (predsig - jnp.mean(predsig)) / jnp.mean(predsig)

    return predsig



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


def penalized_loss(params, data, stimulus, penalty_weight=1e3):
    pred = generate_grid_prediction_jax(params, stimulus)
    mse = jnp.mean((data - pred) ** 2)
    r = jnp.sqrt(params[0]**2 + params[1]**2)
    max_deg = stimulus.deg_x0.max()
    # Nonlinear constraints as penalties
    c1 = jnp.maximum(0.0, r - 2*max_deg)
    c2 = jnp.maximum(0.0, r - 2*params[2] - max_deg)
    penalty = penalty_weight * (c1**2 + c2**2)
    return mse + penalty

def FinalFit_Vox_GPU(init_estim, param_width, timeseries_data, stimulus, use_gpu):
    x0, y0, sigma0, n0 = init_estim[5], init_estim[6], init_estim[3], init_estim[4]
    init_params = jnp.array([x0, y0, sigma0, n0], dtype=jnp.float32)
    max_deg = float(stimulus.deg_x0.max())
    bounds = jnp.array([
        [-2*max_deg, 2*max_deg],
        [-2*max_deg, 2*max_deg],
        [0.001, 2*max_deg],
        [0.001, 2.0]
    ], dtype=jnp.float32)
    # Prepare optimizer
    solver = LBFGSB(fun=lambda p: penalized_loss(p, timeseries_data, stimulus),
                    lower=bounds[:,0], upper=bounds[:, 1], tol=1e-6, maxiter=100)
    sol = solver.run(init_params)
    p_opt = sol.params
    # Compute final fit metrics (can use overload_estimate logic here)
    # (You'll likely want to move this to JAX as well for speed)
    # return jax.device_get(p_opt)
    prediction = np.array(generate_grid_prediction_jax(p_opt, stimulus))
    # Compute overload estimate
    overload_finestim = overload_estimate(p_opt, timeseries_data, prediction)

    # Compare r² to initial estimate
    best_r2 = init_estim[1]
    best_fit = init_estim
    if overload_finestim[1] > best_r2:
        best_fit = overload_finestim

    return best_fit


def get_final_estims_batch(gFit, param_width, timeseries_data, stimulus, fFit, indices, use_gpu=True):
    nvoxs = len(timeseries_data)
    # Prepare args for all voxels
    init_estims = gFit[indices, :]
    # Vectorize FinalFit_Vox_GPU (must be written to accept arrays)
    vmapped_fit = jax.vmap(FinalFit_Vox_GPU, in_axes=(0, None, 0, None, None))
    # Call in batch
    results = vmapped_fit(init_estims, param_width, timeseries_data, stimulus, use_gpu)
    # Store results
    fFit[indices, :] = np.array(results)
    return fFit
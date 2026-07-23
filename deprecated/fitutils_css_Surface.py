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

from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries, generate_rf_timeseries_nomask
import popeye.utilities_cclab as utils

from fit_utils import *


def generate_grid_prediction(args):
    try:
        x, y, sigma, n, stimulus = args
        rf = generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
        # rf = utils.generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
        rf /= ((2 * np.pi * sigma**2) * 1/np.diff(stimulus.deg_x[0,0:2])**2)

        # Extract the stimulus time-series
        response = generate_rf_timeseries_nomask(stimulus.stim_arr, rf)
        # response = utils.generate_rf_timeseries(stimulus.stim_arr, rf)
        response **= n
        predsig = fftconvolve(response, utils.double_gamma_hrf(0, 1.3))[0:len(response)]
        # Normalize the units
        predsig = (predsig - np.mean(predsig)) / np.mean(predsig)

        return predsig
    except Exception as e:
        print(f"Error in generate_grid_prediction: {e}")
        return None

def getGridPreds(grid_space, stimulus, gridPath, nTRs):
    grid_preds = np.empty((len(grid_space), nTRs))#, dtype='float16')
    print("Starting prediction generation")
    with Pool(cpu_count()) as pool:
        results = pool.map(generate_grid_prediction, [(x, y, s, n, stimulus) for x, y, s, n in grid_space])

    for i, prediction in enumerate(results):
        grid_preds[i] = prediction
    # Save grid_preds to disk
    np.save(gridPath, grid_preds)
    return grid_preds

# @numba.jit(nopython=True, parallel=True)
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

def compute_rmse(args):
    data, predictor_series, use_gpu = args
    predictor_series = predictor_series.reshape(-1, 1)
    y = data
    if use_gpu:
        # Raise error that GPU is not supported
        print("GPU not supported for this function. Please set use_gpu=False")
    else:
        X = np.hstack((np.ones((predictor_series.shape[0], 1)), predictor_series))
        XtX = np.dot(X.T, X)
        XtX_inv = np.linalg.inv(XtX)
        XtX_inv_Xt = np.dot(XtX_inv, X.T)
        betas = np.dot(XtX_inv_Xt, y)

        predictions = np.dot(X, betas)

        rmse = np.mean((data - predictions)**2)        
        return rmse

def process_voxel(args):
    timeseries_data, grid_preds, grid_space, use_gpu = args
    ngrids = len(grid_preds)
    
    args = [(timeseries_data, grid_preds[j], use_gpu) for j in range(ngrids)]

    if use_gpu:
        # Raise error that GPU is not supported
        print("GPU not supported for this function. Please set use_gpu=False")
    else:
        rmses = np.array([compute_rmse(arg) for arg in args])
        best_grid_idx = np.argmin(rmses)
        best_grid_estim = grid_space[best_grid_idx]
        best_grid_pred = grid_preds[best_grid_idx]
    overload_estim = overload_estimate(best_grid_estim, timeseries_data, best_grid_pred, use_gpu)
    
    return overload_estim

def get_grid_estims(grid_preds, grid_space, timeseries_data, gFit, indices, use_gpu=False):
    
    nvoxs = len(timeseries_data)
    if use_gpu:
        # Raise error that GPU is not supported
        print("GPU not supported for this function. Please set use_gpu=False")
    else:
        timeseries_data = utils.generate_shared_array(timeseries_data, ctypes.c_double)
        grid_preds = utils.generate_shared_array(grid_preds, ctypes.c_double)
        args = [(timeseries_data[iin, :], grid_preds, grid_space, use_gpu) for iin in range(nvoxs)]
        
        with Pool(cpu_count()) as pool:
            results = []
            for result in tqdm(pool.imap(process_voxel, args), total=nvoxs, dynamic_ncols=False):
                results.append(result)
        
        for i, result in enumerate(results):
            gFit[indices[i], :] = result
    
    return gFit



def FinalFit_Vox(args):
    init_estim, param_width, timeseries_data, stimulus, use_gpu = args
    x_estim, y_estim, sigma_estim, n_estim = init_estim[5], init_estim[6], init_estim[3], init_estim[4]
    beta_estim, baseline_estim = init_estim[7], init_estim[8]
    
    # Define bounds based on initial estimate from grid-fit
    # bounds = generate_bounds(init_estim, param_width)
    bounds = ((-stimulus.deg_x0.max()*2, stimulus.deg_x0.max()*2),
                (-stimulus.deg_y0.max()*2, stimulus.deg_y0.max()*2),
                (0.001, stimulus.deg_x0.max()*2),
                (0.001, 2))
    if np.isnan(timeseries_data).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    unscaled_data = (timeseries_data - baseline_estim) / beta_estim
    
    constraints = (
            NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 
                                -np.inf, 2*stimulus.deg_x0.max()),
            NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2) - 2*x[2], 
                                -np.inf, stimulus.deg_x0.max())
        )
    
    nIters = 1
    widthBuff = 3

    # Initialize best params
    best_r2 = init_estim[1]
    best_fit = init_estim

    # for _ in range(nIters):
    # x_guess = np.random.uniform(x_estim-widthBuff*param_width[0], x_estim+widthBuff*param_width[0])
    # y_guess = np.random.uniform(y_estim-widthBuff*param_width[1], y_estim+widthBuff*param_width[1])
    # sigma_guess = np.random.uniform(sigma_estim-widthBuff*param_width[2], sigma_estim+widthBuff*param_width[2])
    # n_guess = np.random.uniform(n_estim-widthBuff*param_width[3], n_estim+widthBuff*param_width[3])
    x_guess, y_guess, sigma_guess, n_guess = x_estim, y_estim, sigma_estim, n_estim

    try:
        finfit = minimize(error_func,
                        [x_guess, y_guess, sigma_guess, n_guess],
                            bounds=bounds,
                            method = 'SLSQP',
                            args=(unscaled_data, stimulus, generate_grid_prediction),
                            constraints = constraints)#,
        overload_finestim = overload_estimate(finfit.x, unscaled_data, generate_grid_prediction([*finfit.x, stimulus]))
        if overload_finestim[1] > best_r2:
            best_r2 = overload_finestim[1]
            best_fit = overload_finestim
    except ValueError as e:
        pass
        # continue
    
    return best_fit

def get_final_estims(gFit, param_width, timeseries_data, stimulus, fFit, indices, use_gpu=False):
    timeseries_data = utils.generate_shared_array(timeseries_data, ctypes.c_double)
    gFit = utils.generate_shared_array(gFit, ctypes.c_double)

    fFit = np.empty((timeseries_data.shape[0], 9))
    nvoxs = len(timeseries_data)

    args = [(gFit[indices[iin], :], param_width, timeseries_data[iin, :], stimulus, use_gpu) for iin in range(nvoxs)]
    with Pool(cpu_count()) as pool:
        results = []
        for result in tqdm(pool.imap(FinalFit_Vox, args), total=nvoxs, dynamic_ncols=False):
            results.append(result)
    for i, result in enumerate(results):
        fFit[indices[i], :] = result

    return fFit

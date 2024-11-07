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
    # stimulus, x, y, sigma, n = args
    x, y, sigma, n, stimulus = args
    # Generate RF
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

def getGridPreds(grid_space, stimulus, p, timeseries_data):
    grid_preds = np.empty((len(grid_space), timeseries_data.shape[-1]))#, dtype='float16')
    print("Starting prediction generation")
    with Pool(cpu_count()) as pool:
        results = pool.map(generate_grid_prediction, [(x, y, s, n, stimulus) for x, y, s, n in grid_space])

    for i, prediction in enumerate(results):
        grid_preds[i] = prediction
    # Save grid_preds to disk
    np.save(p['gridfit_path'], grid_preds)
    return grid_preds

# def overload_estimate(estimate, data, prediction):
#       # The input to this should (x, y, sigma, n)
#       # The output should be (theta, r2, rho, sigma, n, x, y, beta, baseline)
#       [beta, baseline] = linregress(prediction, data)[0:2]
#       scaled_prediction = beta * prediction + baseline
#       r2 = np.corrcoef(data, scaled_prediction)[0, 1]**2
#       theta = np.mod(np.arctan2(estimate[1], estimate[0]), 2*np.pi)
#       rho = np.sqrt(estimate[0]**2 + estimate[1]**2)
#       return (theta, r2, rho, estimate[2], estimate[3], estimate[0], estimate[1], beta, baseline)

# def overload_estimate(estimate, data, prediction):
#     X = np.vstack((np.ones(len(prediction)), prediction)).T
#     betas = np.linalg.lstsq(X, data, rcond=None)[0]
#     scaled_prediction = np.dot(X, betas)
#     r2 = np.corrcoef(data, scaled_prediction)[0, 1]**2
#     theta = np.mod(np.arctan2(estimate[1], estimate[0]), 2*np.pi)
#     rho = np.sqrt(estimate[0]**2 + estimate[1]**2)
#     return (theta, r2, rho, estimate[2], estimate[3], estimate[0], estimate[1], betas[1], betas[0])
@cuda.jit
def compute_overload_kernel(X, y, betas, result):
    idx = cuda.grid(1)
    if idx < X.shape[0]:
        prediction = 0.0
        for j in range(X.shape[1]):
            prediction += X[idx, j] * betas[j]
        error = y[idx] - prediction
        result[idx] = error * error
# @cuda.jit
# def compute_rmse_kernel(X, y, betas, rmse_arr):
#     idx = cuda.grid(1)
#     if idx < X.shape[0]:
#         prediction = 0.0
#         for j in range(X.shape[1]):
#             prediction += X[idx, j] * betas[j]
#         error = y[idx] - prediction
#         rmse_arr[idx] = error * error   

def overload_estimate(estimate, data, prediction, use_gpu=False):
    # Returns (theta, r2, rho, sigma, n, x, y, beta, baseline)
    if use_gpu:
        X = cp.vstack((cp.ones(len(prediction), dtype=cp.float32), prediction)).T
        XtX = cp.dot(X.T, X)
        XtY = cp.dot(X.T, data)
        betas = cp.linalg.solve(XtX, XtY)
        scaled_prediction = cp.dot(X, betas)
        r2 = cp.corrcoef(data, scaled_prediction)[0, 1]**2
        theta = cp.mod(cp.arctan2(estimate[1], estimate[0]), 2*cp.pi)
        rho = cp.sqrt(estimate[0]**2 + estimate[1]**2)
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
        X = cp.hstack((cp.ones((predictor_series.shape[0], 1), dtype=cp.float32), predictor_series))
        y = cp.asarray(y, dtype=cp.float32)

        # Compute the betas
        XtX = cp.dot(X.T, X)
        XtX_inv = cp.linalg.inv(XtX)
        XtX_inv_Xt = cp.dot(XtX_inv, X.T)
        betas = cp.dot(XtX_inv_Xt, y)

        # Compute predictions and RMSE
        predictions = cp.dot(X, betas)
        rmse = cp.mean((data - predictions)**2)
        if cp.any(betas[1:] < 0):
            rmse = 1000000
        return cp.asnumpy(rmse)
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
    iin, timeseries_data, grid_preds, grid_space, indices, use_gpu = args
    ngrids = len(grid_preds)
    
    args = [(timeseries_data, grid_preds[j], use_gpu) for j in range(ngrids)]

    if use_gpu:
        rmses = cp.array([compute_rmse(arg) for arg in args])
        best_grid_idx = cp.argmin(rmses)
        best_grid_estim = grid_space[int(cp.asnumpy(best_grid_idx))]
        best_grid_pred = grid_preds[int(cp.asnumpy(best_grid_idx))]
    else:
        rmses = np.array([compute_rmse(arg) for arg in args])
        best_grid_idx = np.argmin(rmses)
        best_grid_estim = grid_space[best_grid_idx]
        best_grid_pred = grid_preds[best_grid_idx]
    overload_estim = overload_estimate(best_grid_estim, timeseries_data, best_grid_pred, use_gpu)
    iix, iiy, iiz = indices[iin]
    
    return iix, iiy, iiz, overload_estim

def get_grid_estims(grid_preds, grid_space, timeseries_data, gFit, indices, use_gpu=False):
    
    nvoxs = len(timeseries_data)
    
    if use_gpu:
        import cupy as cp
        timeseries_data = cp.asarray(timeseries_data, dtype=cp.float32)
        grid_preds = cp.asarray(grid_preds, dtype=cp.float32)
        

        args = [(iin, timeseries_data[iin, :], grid_preds, grid_space, indices, use_gpu) for iin in range(nvoxs)]
        results = [process_voxel(arg) for arg in args]
        for iix, iiy, iiz, overload_estim in results:
            gFit[iix, iiy, iiz, :] = overload_estim

    else:
        timeseries_data = utils.generate_shared_array(timeseries_data, ctypes.c_double)
        grid_preds = utils.generate_shared_array(grid_preds, ctypes.c_double)
        args = [(iin, timeseries_data[iin, :], grid_preds, grid_space, indices, use_gpu) for iin in range(nvoxs)]
        
        with Pool(cpu_count()) as pool:
            results = []
            for result in tqdm(pool.imap(process_voxel, args), total=nvoxs, dynamic_ncols=False):
                results.append(result)
        
        for iix, iiy, iiz, overload_estim in results:
            gFit[iix, iiy, iiz, :] = overload_estim
    
    return gFit

def get_grid2_estims(grid_preds, grid_space, voxel_data):
    ngrids = len(grid_preds)
    rmses = []
    for j in range(ngrids):
        rmses.append(compute_rmse([voxel_data, grid_preds[j]]))
    best_grid_estim = grid_space[np.argmin(rmses)]
    overload_estim = overload_estimate(best_grid_estim, voxel_data, grid_preds[np.argmin(rmses)])
    return overload_estim

def rerun_gFit_vox(args):
    voxel_data, stimulus, param_width, grid1_estim = args
    Ns = 10
    # grid1_estim = gFitorig[idx, idy, idz, :]
    # print(grid1_estim)
    x_estim, y_estim, sigma_estim, n_estim = grid1_estim[5], grid1_estim[6], grid1_estim[3], grid1_estim[4]
    
    # Generate grids for this voxel
    x_grid = np.linspace(x_estim-param_width[0], x_estim+param_width[0], Ns, dtype=np.float32)
    y_grid = np.linspace(y_estim-param_width[1], y_estim+param_width[1], Ns, dtype=np.float32)
    s_grid = np.linspace(sigma_estim-param_width[2], sigma_estim+param_width[2], Ns, dtype=np.float32)
    # n_grid = np.linspace(n_estim-param_width[3], n_estim+param_width[3], Ns)
    n_grid = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    grid_space_orig = list(product(x_grid, y_grid, s_grid, n_grid))
    # print(grid_space_orig)
    # Constraint the grids
    grid_space = constraint_grids(grid_space_orig, stimulus)
    if len(grid_space) > 0:
        grid_preds = np.empty((len(grid_space), voxel_data.shape[-1]))#, dtype='float16')

        for i in range(len(grid_space)):
            grid_preds[i] = generate_grid_prediction((grid_space[i][0], grid_space[i][1], grid_space[i][2], grid_space[i][3], stimulus))
        
        gfit_estim = get_grid2_estims(grid_preds, grid_space, voxel_data)
    else:
        gfit_estim = grid1_estim
    return gfit_estim

def rerun_gridFit_parallel(gFitorig, timeseries_data, stimulus, param_width, gFit, indices, use_gpu=False):
    timeseries_data = utils.generate_shared_array(timeseries_data, ctypes.c_double)
    gFitorig = utils.generate_shared_array(gFitorig, ctypes.c_double)

    nvoxs = len(timeseries_data)
    
    args = [(timeseries_data[iin, :], stimulus, param_width, 
             gFitorig[indices[iin][0], indices[iin][1], indices[iin][2], :]) for iin in range(nvoxs)]
    with Pool(cpu_count()) as pool:
        results = []
        for result in tqdm(pool.imap(rerun_gFit_vox, args), total=nvoxs, dynamic_ncols=False):
            results.append(result)
        # results = pool.map(rerun_gFit_vox, [(timeseries_data[iin, :], stimulus, param_width, 
        #                                      gFitorig[indices[iin][0], indices[iin][1], indices[iin][2], :]) for iin in range(nvoxs)])
    
    for iin, result in enumerate(results):
        iix, iiy, iiz = indices[iin]
        gFit[iix, iiy, iiz, :] = result

    return gFit

def rerun_gridFit(gFitorig, timeseries_data, stimulus, param_width, gFit, indices):
    nvoxs = len(timeseries_data)
    Ns = 10
    for iin in range(nvoxs):
        grid1_estim = gFitorig[indices[iin][0], indices[iin][1], indices[iin][2], :]
        x_estim, y_estim, sigma_estim, n_estim = grid1_estim[5], grid1_estim[6], grid1_estim[3], grid1_estim[4]
        
        # Generate grids for this voxel
        x_grid = np.linspace(x_estim-param_width[0], x_estim+param_width[0], Ns)
        y_grid = np.linspace(y_estim-param_width[1], y_estim+param_width[1], Ns)
        s_grid = np.linspace(sigma_estim-param_width[2], sigma_estim+param_width[2], Ns)
        n_grid = np.linspace(n_estim-param_width[3], n_estim+param_width[3], Ns)
        grid_space_orig = list(product(x_grid, y_grid, s_grid, n_grid))
        # Constraint the grids
        grid_space = constraint_grids(grid_space_orig, stimulus)

        if len(grid_space) > 0:
            grid_preds = np.empty((len(grid_space), timeseries_data.shape[-1]))#, dtype='float16')
        
            with ThreadPoolExecutor() as executor:
                results = executor.map(generate_grid_prediction, [(x, y, s, n, stimulus) for x, y, s, n in grid_space])
            for i, prediction in enumerate(results):
                grid_preds[i] = prediction

            gFit[indices[iin][0], indices[iin][1], indices[iin][2], :] = get_grid2_estims(grid_preds, grid_space, timeseries_data[iin, :])
        else:
            gFit[indices[iin][0], indices[iin][1], indices[iin][2], :] = grid1_estim
    # gFit = get_grid_estims(grid_preds, grid_space, timeseries_data, gFit, indices)
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
    try:
        finfit = minimize(error_func,
                        [x_estim, y_estim, sigma_estim, n_estim],
                            bounds=bounds,
                            method = 'SLSQP',
                            args=(unscaled_data, stimulus, generate_grid_prediction),
                            constraints = constraints)#,
        overload_finestim = overload_estimate(finfit.x, unscaled_data, generate_grid_prediction([*finfit.x, stimulus]))
        return overload_finestim
    except ValueError as e:
        return init_estim


def get_final_estims(gFit, param_width, timeseries_data, stimulus, fFit, indices, use_gpu=False):
    timeseries_data = utils.generate_shared_array(timeseries_data, ctypes.c_double)
    gFit = utils.generate_shared_array(gFit, ctypes.c_double)
    nvoxs = len(timeseries_data)

    args = [(gFit[indices[iin][0], indices[iin][1], indices[iin][2], :], param_width, timeseries_data[iin, :], stimulus, use_gpu) for iin in range(nvoxs)]
    with Pool(cpu_count()) as pool:
        results = []
        for result in tqdm(pool.imap(FinalFit_Vox, args), total=nvoxs, dynamic_ncols=False):
            results.append(result)
    
    for iin, result in enumerate(results):
        iix, iiy, iiz = indices[iin]
        fFit[iix, iiy, iiz, :] = result
    return fFit
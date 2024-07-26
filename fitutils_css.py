import numpy as np
# import cupy as cp
# from cupyx.scipy.signal import fftconvolve
# from cupyx.scipy.stats import linregress
import torch
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries, generate_rf_timeseries_nomask
import popeye.utilities_cclab as utils
from scipy.signal import fftconvolve
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize, NonlinearConstraint
from itertools import product
from fit_utils import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(device)


def generate_grid_prediction(args):
    # stimulus, x, y, sigma, n = args
    x, y, sigma, n, stimulus = args
    # Generate RF
    rf = generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
    rf /= ((2 * np.pi * sigma**2) * 1/np.diff(stimulus.deg_x[0,0:2])**2)

    # Extract the stimulus time-series
    response = generate_rf_timeseries_nomask(stimulus.stim_arr, rf)
    response **= n
    predsig = fftconvolve(response, utils.double_gamma_hrf(0, 1.3))[0:len(response)]
    # Normalize the units
    predsig = (predsig - np.mean(predsig)) / np.mean(predsig)
    # response_pt = torch.tensor(response, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # hrf = torch.tensor(utils.double_gamma_hrf(0, 1.3), device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # predsig = torch.nn.functional.conv1d(response_pt, hrf, padding='same').squeeze().cpu().numpy()

    # predsig = (predsig - np.mean(predsig)) / np.mean(predsig)

    return predsig

def generate_grid_prediction_new(x, y, sigma, n, stimulus):
    # stimulus, x, y, sigma, n = args
    # x, y, sigma, n, stimulus = args
    # Generate RF
    rf = generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
    rf /= ((2 * np.pi * sigma**2) * 1/np.diff(stimulus.deg_x[0,0:2])**2)

    # Extract the stimulus time-series
    response = generate_rf_timeseries_nomask(stimulus.stim_arr, rf)
    response **= n

    predsig = fftconvolve(response, utils.double_gamma_hrf(0, 1.3))[0:len(response)]
    # Normalize the units
    predsig = (predsig - np.mean(predsig)) / np.mean(predsig)

    # predsig *= beta
    # predsig += baseline
    # response_pt = torch.tensor(response, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # hrf = torch.tensor(utils.double_gamma_hrf(0, 1.3), device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # predsig = torch.nn.functional.conv1d(response_pt, hrf, padding='same').squeeze().cpu().numpy()

    # predsig = (predsig - np.mean(predsig)) / np.mean(predsig)

    return predsig


def overload_estimate(estimate, data, prediction):
      # The input to this should (x, y, sigma, n)
      # The output should be (theta, r2, rho, sigma, n, x, y, beta, baseline)
      [beta, baseline] = linregress(prediction, data)[0:2]
      scaled_prediction = beta * prediction + baseline
      r2 = np.corrcoef(data, scaled_prediction)[0, 1]**2
      theta = np.mod(np.arctan2(estimate[1], estimate[0]), 2*np.pi)
      rho = np.sqrt(estimate[0]**2 + estimate[1]**2)
      return (theta, r2, rho, estimate[2], estimate[3], estimate[0], estimate[1], beta, baseline)
      
def compute_rmse(args):
    data, predictor_series = args
    predictor_series = predictor_series.reshape(-1, 1)
    model = LinearRegression().fit(predictor_series, data)
    predictions = model.predict(predictor_series)
    rmse = mean_squared_error(data, predictions, squared=True)
    # for model betas that are negative, make rmses very large
    for i in range(len(model.coef_)):
        if model.coef_[i] < 0:
            rmse = 1000000
    return rmse

def compute_rmse_gpu(args):
    data, predictor_series = args

    data_pt = torch.tensor(data, device=device, dtype=torch.float32).unsqueeze(1)
    predictor_series_pt = torch.tensor(predictor_series, device=device, dtype=torch.float32).unsqueeze(1)

    model = torch.nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    for _ in range(100):
        optimizer.zero_grad()
        outputs = model(predictor_series_pt)
        loss = criterion(outputs, data_pt)
        loss.backward()
        optimizer.step()

    predictions = model(predictor_series_pt).cpu().detach().numpy()
    rmse = mean_squared_error(data, predictions, squared=True)

    if model.weight.item() < 0:
        rmse = 1000000
    return rmse

def process_voxel(args):
    iin, timeseries_data, grid_preds, grid_space, indices = args
    ngrids = len(grid_preds)
    
    args = [(timeseries_data[iin, :], grid_preds[j]) for j in range(ngrids)]

    rmses = []
    for j in range(ngrids):
        rmses.append(compute_rmse(args[j]))
    
    best_grid_estim = grid_space[np.argmin(rmses)]
    overload_estim = overload_estimate(best_grid_estim, timeseries_data[iin, :], grid_preds[np.argmin(rmses)])
    iix, iiy, iiz = indices[iin]
    
    return iix, iiy, iiz, overload_estim

def get_grid_estims(grid_preds, grid_space, timeseries_data, gFit, indices):
    nvoxs = len(timeseries_data)
    
    args = [(iin, timeseries_data, grid_preds, grid_space, indices) for iin in range(nvoxs)]
    
    with Pool(cpu_count()) as pool:
        results = pool.map(process_voxel, args)
    # with ThreadPoolExecutor() as executor:
    #     results = executor.map(process_voxel, args)
    
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
    Ns = 5
    # grid1_estim = gFitorig[idx, idy, idz, :]
    # print(grid1_estim)
    x_estim, y_estim, sigma_estim, n_estim = grid1_estim[5], grid1_estim[6], grid1_estim[3], grid1_estim[4]
    
    # Generate grids for this voxel
    x_grid = np.linspace(x_estim-param_width[0], x_estim+param_width[0], Ns)
    y_grid = np.linspace(y_estim-param_width[1], y_estim+param_width[1], Ns)
    s_grid = np.linspace(sigma_estim-param_width[2], sigma_estim+param_width[2], Ns)
    # n_grid = np.linspace(n_estim-param_width[3], n_estim+param_width[3], Ns)
    n_grid = np.asarray([0.25, 0.5, 0.75, 1.0])
    grid_space_orig = list(product(x_grid, y_grid, s_grid, n_grid))
    # print(grid_space_orig)
    # Constraint the grids
    grid_space = constraint_grids(grid_space_orig, stimulus)
    if len(grid_space) > 0:
        grid_preds = np.empty((len(grid_space), voxel_data.shape[-1]))#, dtype='float16')

        for i in range(len(grid_space)):
            grid_preds[i] = generate_grid_prediction((grid_space[i][0], grid_space[i][1], grid_space[i][2], grid_space[i][3], stimulus))
        
        # with ThreadPoolExecutor() as executor:
        #     results = executor.map(generate_grid_prediction, [(x, y, s, n, stimulus) for x, y, s, n in grid_space])
        # for i, prediction in enumerate(results):
        #     grid_preds[i] = prediction
        gfit_estim = get_grid2_estims(grid_preds, grid_space, voxel_data)
    else:
        gfit_estim = grid1_estim
    return gfit_estim

def rerun_gridFit_parallel(gFitorig, timeseries_data, stimulus, param_width, gFit, indices):
    nvoxs = len(timeseries_data)

    with Pool(cpu_count()) as pool:
        results = pool.map(rerun_gFit_vox, [(timeseries_data[iin, :], stimulus, param_width, 
                                             gFitorig[indices[iin][0], indices[iin][1], indices[iin][2], :]) for iin in range(nvoxs)])
    
    for iin, result in enumerate(results):
        iix, iiy, iiz = indices[iin]
        gFit[iix, iiy, iiz, :] = result

    return gFit

def rerun_gridFit(gFitorig, timeseries_data, stimulus, param_width, gFit, indices):
    nvoxs = len(timeseries_data)
    Ns = 3
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
    
     
def get_final_estims(gFit, param_width, timeseries_data, stimulus, fFit, indices):
    nvoxs = len(timeseries_data)

    for iin in range(nvoxs):
        init_estim = gFit[indices[iin][0], indices[iin][1], indices[iin][2], :]
        x_estim, y_estim, sigma_estim, n_estim = init_estim[5], init_estim[6], init_estim[3], init_estim[4]
        beta_estim, baseline_estim = init_estim[7], init_estim[8]

        bounds = generate_bounds(init_estim, param_width)
        # Define bounds based on initial estimate from grid-fit
        

        unscaled_data = (timeseries_data[iin, :] - baseline_estim) / beta_estim
        # finfit = minimize(error_func,
        #                   [x_estim, y_estim, sigma_estim, n_estim],
        #                   bounds=bounds,
        #                   method = 'SLSQP',
        #                     # method='COBYLA',
        #                   args=(unscaled_data, stimulus, generate_grid_prediction_new))#,
                        #   constraints = ({'type': 'ineq', 'fun': lambda x: np.sqrt(x[0]**2 + x[1]**2) - 2*stimulus.deg_x0.max()},
                        #                   {'type': 'ineq', 'fun': lambda x: np.sqrt(x[0]**2 + x[1]**2) - (stimulus.deg_x0.max() + 2*x[2])}))
                        #   constraints = ({'type': 'ineq', 'fun': lambda x: 2*stimulus.deg_x0.max() - np.sqrt(x[0]**2 + x[1]**2)},
                        #                     {'type': 'ineq', 'fun': lambda x: stimulus.deg_x0.max() + 2*x[2] - np.sqrt(x[0]**2 + x[1]**2)}))
                        
        constraints = (
                NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2), 
                                    -np.inf, 2*stimulus.deg_x0.max()),
                NonlinearConstraint(lambda x: np.sqrt(x[0]**2 + x[1]**2) - 2*x[2], 
                                    -np.inf, stimulus.deg_x0.max())
            )
        finfit = minimize(error_func,
                          [x_estim, y_estim, sigma_estim, n_estim],
                          bounds=bounds,
                          method = 'COBYLA',
                          args=(unscaled_data, stimulus, generate_grid_prediction),
                          constraints = constraints)#,
        overload_finestim = overload_estimate(finfit.x, unscaled_data, generate_grid_prediction([*finfit.x, stimulus]))
        # overload_finestim = overload_estimate(finfit.x, timeseries_data[iin, :], generate_grid_prediction([*finfit.x, stimulus]))
        iix, iiy, iiz = indices[iin]
        fFit[iix, iiy, iiz, :] = overload_finestim
    return fFit
    
    

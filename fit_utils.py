import numpy as np
import ctypes, time, os
import matplotlib.pyplot as plt
from scipy.signal import detrend
from sklearn.metrics import mean_squared_error
# Import popeye stuff
import popeye.utilities_cclab as utils
import popeye.models_cclab as prfModels

import multiprocessing as mp

def print_time(st_time, end_time, process_name):
    duration = end_time - st_time
    if duration < 60:
        print(f'{process_name} took {round(duration, 2)} seconds')
    elif duration < 3600:
        # print mins:secs
        print(f'{process_name} took {round(duration/60)} minutes and {round(duration%60)} seconds')
        # print(f'{process_name} took {round(duration/60, 2)} minutes')
    else:
        # print hours:mins:secs
        print(f'{process_name} took {round(duration//3600)} hours, {round((duration%3600)//60)} minutes and {round(duration%60)} seconds')
        # print(f'{process_name} took {round(duration/3600, 2)} hours')

def remove_trend(signal, method='all'):
    if method == 'demean':
         return (signal - np.mean(signal, axis=-1)[..., None]) / np.mean(signal, axis=-1)[..., None]
    elif method == 'prct_signal_change':
        return utils.percent_change(signal, ax=-1)
    elif method == 'all':
        signal_mean = np.mean(signal, axis=-1)[..., None]
        signal_detrend = detrend(signal, axis=-1, type='linear') + signal_mean
        signal_pct = utils.percent_change(signal_detrend, ax=-1)
        return signal_pct

# def constraint_grids(grid_space_orig, stimulus):
#     # print(f'Number of grid points: {len(grid_space_orig)}')
#     idxs_to_drop = []
#     for i in range(len(grid_space_orig)):
#         if np.sqrt(grid_space_orig[i][0]**2 + grid_space_orig[i][1]**2) >= 2*stimulus.deg_x0.max():
#             idxs_to_drop.append(i)
#         if np.sqrt(grid_space_orig[i][0]**2 + grid_space_orig[i][1]**2) >= stimulus.deg_x0.max() + 2*grid_space_orig[i][2]:
#             idxs_to_drop.append(i)
#     grid_space = [grid_space_orig[i] for i in range(len(grid_space_orig)) if i not in idxs_to_drop]
#     # print(f'Number of grid points after dropping: {len(grid_space)}')
#     return grid_space

def constraint_grids(grid_space_orig, stimulus):
    grid_space_orig = np.array(grid_space_orig)
    x, y, s, n = grid_space_orig[:, 0], grid_space_orig[:, 1], grid_space_orig[:, 2], grid_space_orig[:, 3]
    distances = np.sqrt(x**2 + y**2)
    # Define constraints
    max_dist1 = 2*stimulus.deg_x0.max()
    max_dist2 = stimulus.deg_x0.max() + 2*s
    # Create a boolean mask
    mask = (distances < max_dist1) & (distances < max_dist2)
    # Apply the mask
    grid_space = grid_space_orig[mask]
    return grid_space.tolist()


def generate_bounds(init_estim, param_width):
    x_estim, y_estim, sigma_estim, n_estim = init_estim[5], init_estim[6], init_estim[3], init_estim[4]
    
    [x_bound_min, x_bound_max] = [x_estim - param_width[0], x_estim + param_width[0]]
    [y_bound_min, y_bound_max] = [y_estim - param_width[1], y_estim + param_width[1]]
    [sigma_bound_min, sigma_bound_max] = [sigma_estim - param_width[2], sigma_estim + param_width[2]]
    [n_bound_min, n_bound_max] = [n_estim - param_width[3], n_estim + param_width[3]]
    x_bounds = (x_bound_min, x_bound_max)
    y_bounds = (y_bound_min, y_bound_max)
    sigma_bounds = (sigma_bound_min, sigma_bound_max)
    n_bounds = (n_bound_min, n_bound_max)

    # print(iin, x_bounds, y_bounds, sigma_bounds, n_bounds)
    
    bounds = (x_bounds, y_bounds, sigma_bounds, n_bounds)
    return bounds

def error_func(parameters, data, stimulus, objective_function):
    # prediction = objective_function(*parameters, stimulus)
    prediction = objective_function([*parameters, stimulus])
    error = mean_squared_error(data, prediction, squared=True)
    return error
import numpy as np
import ctypes, time, os
import matplotlib.pyplot as plt
from scipy.signal import detrend


# Import popeye stuff
import popeye.utilities_cclab as utils
import popeye.models_cclab as prfModels

import multiprocessing as mp

def run_fit(index, scan_data_visual, css_model, grids, bounds, auto_fit, verbose):
    return fit_voxel(index, scan_data_visual, css_model, grids, bounds, auto_fit, verbose)

def fit_voxel(args):
    # ix, iy, iz, model, data, grids, bounds, auto_fit=True, grid_only=False, verbose=0, visual_rois=None, vx_indices=None, start_time=None):
    ix, iy, iz, model, data, grids, bounds, auto_fit, grid_only, verbose, visual_rois, vx_indices, start_time = args
    if visual_rois[ix, iy, iz] == 1:
        th_vx_idx = np.where((vx_indices == (ix, iy, iz)).all(axis=1))[0][0]
        if np.mod(th_vx_idx, 100) == 0:
            run_time = time.time() - start_time
            if run_time < 60:
                print(f"Finished: {round(th_vx_idx/len(vx_indices)*100, 2)}%, time: {round(run_time, 2)} s")
            elif run_time < 3600:
                print(f"Finished: {round(th_vx_idx/len(vx_indices)*100, 2)}%, time: {int(np.floor(run_time/60))} min {round(run_time%60)} s")
            else:
                print(f"Finished: {round(th_vx_idx/len(vx_indices)*100, 2)}%, time: {int(np.floor(run_time/3600))} h {int(np.floor(run_time%3600/60))} min {round(run_time%60)} s")
        voxel_data = data[ix, iy, iz, :]
        fit = prfModels.CompressiveSpatialSummationFit(
            model,
            voxel_data,
            grids,
            bounds,
            (ix, iy, iz),
            auto_fit=auto_fit,
            grid_only=grid_only,
            verbose=verbose
        )
        return (ix, iy, iz, fit.theta0, fit.rsquared0, fit.rho0, fit.s0, fit.n0, fit.x0, fit.y0, fit.beta0, fit.baseline0,
                            fit.theta, fit.rsquared, fit.rho, fit.sigma, fit.n, fit.x, fit.y, fit.beta, fit.baseline)
    
    return None

def remove_trend(signal, method='all'):
    if method == 'demean':
         return (signal - np.mean(signal, axis=-1)[..., None]) / np.mean(signal, axis=-1)[..., None]
        #return detrend(scan_data, axis=-1, type='constant')
    # elif method == 'detrend':
    #     mean_signal = np.mean(signal, axis=-1)[..., None]
    #     return detrend(signal, axis=-1) + mean_signal
    elif method == 'prct_signal_change':
        return utils.percent_change(signal, ax=-1)
    elif method == 'all':
        signal_mean = np.mean(signal, axis=-1)[..., None]
        signal_detrend = detrend(signal, axis=-1, type='linear') + signal_mean
        signal_pct = utils.percent_change(signal_detrend, ax=-1)
        return signal_pct
#         timeseries_data_mean = np.mean(timeseries_data, axis=-1)
# timeseries_data_detrend = detrend(timeseries_data, axis=-1, type='linear') + timeseries_data_mean[:, np.newaxis]
# timeseries_data_pct = utils.percent_change(timeseries_data_detrend, ax=-1)
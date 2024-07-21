import numpy as np
import ctypes, time, os
import matplotlib.pyplot as plt

# Import popeye stuff
import popeye.utilities_cclab as utils
from popeye.visual_stimulus import VisualStimulus
import popeye.models_cclab as prfModels

import multiprocessing as mp

def fit_voxel(index, scan_data_visual, css_model, grids, bounds, auto_fit, verbose):
    ix, iy, iz = index
    voxel_data = scan_data_visual[ix, iy, iz, :]

    fit_result = css.CompressiveSpatialSummationFit(
        css_model,
        voxel_data,
        grids=grids,
        bounds=bounds,
        voxel_index=index,
        Ns=1,
        auto_fit=auto_fit,
        verbose=verbose
    )
    
    return fit_result

def run_fit(index, scan_data_visual, css_model, grids, bounds, auto_fit, verbose):
    return fit_voxel(index, scan_data_visual, css_model, grids, bounds, auto_fit, verbose)

def process_voxel(args):
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

def remove_trend(signal, method='demean'):
    if method == 'demean':
         return (signal - np.mean(signal, axis=-1)[..., None]) / np.mean(signal, axis=-1)[..., None]
        #return detrend(scan_data, axis=-1, type='constant')
    # elif method == 'detrend':
    #     mean_signal = np.mean(signal, axis=-1)[..., None]
    #     return detrend(signal, axis=-1) + mean_signal
    elif method == 'prct_signal_change':
        return utils.percent_change(signal, ax=-1)
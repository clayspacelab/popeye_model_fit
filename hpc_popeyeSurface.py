import numpy as np
import ctypes, time, os
import sys
# from ipywidgets import interact, widgets

# Import visualization stuff
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from scipy.signal import detrend

# Import popeye stuff
import popeye.utilities_cclab as utils
from popeye.visual_stimulus import VisualStimulus
# import popeye.models_cclab as prfModels
# import popeye.css_cclab as prfModels

# Import multiprocessing stuff
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import product

# Load helper functions
from helpersSurface import set_paths_surface, load_stimuli, averageRuns, save2gifti
from fit_utils import *
from fitutils_css_Surface import *
import ctypes

def main():
    if len(sys.argv) != 2:
        print("Usage: python popeyeFitter.py subjID")
        sys.exit(1)

    # gridfit2
    # Initialize parameters
    print('Initializing parameters ...')
    codeStartTime = time.perf_counter()
    params = {}
    params['subjID'] = sys.argv[1]
    # Got these from Zhengang, and he got it from rsvp_params.txt
    params['viewingDistance'] = 83.5 #63 #83.5 # in cm
    params['screenWidth'] = 36.2 #35 #36.2 # in cm
    params['scaleFactor'] = 1
    params['resampleFactor'] = 1
    params['dtype'] = ctypes.c_int16

    p, funcFiles = set_paths_surface(params)

    # Load stimulus
    bar, _ = load_stimuli(p)
    # bar = bar[:, :, 0:201]
    # Mirror y axis (this is done because popeye flips the y axis)
    bar = np.flip(bar, axis=0)

    # Average runs and get metadata
    leftDataOrig, rightDataOrig, tr_length, nTRs = averageRuns(p, funcFiles)
    params['tr_length'] = tr_length
    params['nTRs'] = nTRs

    # load scan data
    print('Detrending scan data ...')
    leftDataOrig = remove_trend(leftDataOrig, method='all')
    rightDataOrig = remove_trend(rightDataOrig, method='all')
    

    # nvoxs = leftDataOrig.shape[0]
    # print(f"Running model-fit on {nvoxs} voxels for each hemisphere")
    # Select 100 random voxels from brainmask
    voxelsLeft = np.arange(leftDataOrig.shape[0])
    # np.random.shuffle(voxelsLeft)
    indicesLeft = voxelsLeft#[:nvoxs]
    voxelsRight = np.arange(rightDataOrig.shape[0])
    # np.random.shuffle(voxelsRight)
    indicesRight = voxelsRight#[:nvoxs]
    leftData = leftDataOrig#[indicesLeft, :]
    rightData = rightDataOrig#[indicesRight, :]


    # create stimulus object from popeye
    print('Creating stimulus object ...')
    stimulus = VisualStimulus(bar.astype('int16'),
                            params['viewingDistance'],
                            params['screenWidth'],
                            params['scaleFactor'],
                            params['tr_length'],
                            params['dtype'],
    )

    # set search grids
    Ns = 35
    x_grid = np.concatenate((np.linspace(-stimulus.deg_x0.max(), stimulus.deg_x0.max(), Ns//2),
                        np.geomspace(-stimulus.deg_x0.max(), -2*stimulus.deg_x0.max(), Ns//4),
                            np.geomspace(stimulus.deg_x0.max(), 2*stimulus.deg_x0.max(), Ns//4)))
    y_grid = np.concatenate((np.linspace(-stimulus.deg_y0.max(), stimulus.deg_y0.max(), Ns//2),
                            np.geomspace(-stimulus.deg_y0.max(), -2*stimulus.deg_y0.max(), Ns//4),
                            np.geomspace(stimulus.deg_y0.max(), 2*stimulus.deg_y0.max(), Ns//4)))
    s_grid = np.concatenate((np.linspace(0.1, 5, 3*Ns//4), np.geomspace(5, stimulus.deg_x0.max(), Ns//4)))
    n_grid = np.asarray([0.25, 0.5, 0.75, 1])
    grid_space_orig = list(product(x_grid, y_grid, s_grid, n_grid))
    grid_space = constraint_grids(grid_space_orig, stimulus)
    print(f'Number of grid points: {len(grid_space)}')

    param_width = [np.mean(np.diff(x_grid)), np.mean(np.diff(y_grid)), np.mean(np.diff(s_grid)), np.mean(np.diff(n_grid))]
    # param_width = np.asarray(round(param_width, 4))
    
    tstamp_start = time.perf_counter()
    
    ############################  GRID PREDICTIONS ################################
    if Ns == 25:
        gridPath = p['gridfit_path_25']
    elif Ns == 35:
        gridPath = p['gridfit_path_35']
    elif Ns == 50:
        gridPath = p['gridfit_path_50']
    if os.path.exists(gridPath):
        print("Loading grid predictions from disk")
        grid_preds = np.load(gridPath)
    else:
        print("Grid predictions don't exist. Generating them")
        grid_preds = getGridPreds(grid_space, stimulus, gridPath, nTRs)

    tstamp_gridpred = time.perf_counter()
    # print(f'Obtained grid predictions in {tstamp_gridpred-tstamp_start} seconds')
    print_time(tstamp_start, tstamp_gridpred, 'Grid predictions')
    
    ############################  GRID FIT ################################
    
    if os.path.exists(os.path.join(p['fitEstimDir'], 'RF_ss5_gFit_popeye_left.func.gii')):
        print("Loading grid estimates from disk")
        # RF_left_gFit_img = nib.load(os.path.join(p['fitEstimDir'], 'RF_ss5_gFit_popeye_left.func.gii'))
        # RF_left_gFit = np.array([x.data for x in RF_left_gFit_img.darrays]).T
        # print(RF_left_gFit.shape)
        RF_right_gFit_img = nib.load(os.path.join(p['fitEstimDir'], 'RF_ss5_gFit_popeye_right.func.gii'))
        RF_right_gFit = np.array([x.data for x in RF_right_gFit_img.darrays]).T
        print(RF_right_gFit.shape)
    else:
        print('Starting grid fit ...')
        # RF_left_gFit = np.empty((leftDataOrig.shape[0], 9))
        RF_right_gFit = np.empty((rightDataOrig.shape[0], 9))
        # RF_left_gFit = get_grid_estims(grid_preds, grid_space, leftData, RF_left_gFit, indicesLeft)
        RF_right_gFit = get_grid_estims(grid_preds, grid_space, rightData, RF_right_gFit, indicesRight)
        tstamp_gridestim = time.perf_counter()
        print(f'Obtained grid estimates in {tstamp_gridestim-tstamp_gridpred} seconds')
        print_time(tstamp_gridpred, tstamp_gridestim, 'Grid fit1')

        # Save the results
        # save2gifti(RF_left_gFit, fpath=os.path.join(p['fitEstimDir'], 'RF_ss5_gFit_popeye_left.func.gii'))
        save2gifti(RF_right_gFit, fpath=os.path.join(p['fitEstimDir'], 'RF_ss5_gFit_popeye_right.func.gii'))

        # f0, axs = plt.subplots(2, 4, figsize=(20, 10))
        # axs = axs.flatten()
        # for i in range(8):
        #     ax = axs[i]
        #     ax.hist(RF_left_gFit[indicesLeft, i].flatten(), bins=50, alpha=0.5, label='Left')
        #     # ax.hist(RF_right_gFit[indicesRight, i].flatten(), bins=50, alpha=0.5, label='Right')
        #     ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
        #     ax.set_title(f"Grid-fit: {['theta', 'rsquared', 'rho', 'sigma','n', 'x', 'y', 'beta'][i]}")
        #     ax.set_xlabel('mrVista')
        #     ax.set_ylabel('Popeye')
        # plt.savefig(os.path.join(p['fig_dir'], 'gridfit_comparison.png'), dpi=300)
        # plt.close(f0)

    ############################  GRID FIT2 ################################
    # RF_ss5_g2Fit = np.empty((scan_data.shape[0], scan_data.shape[1], scan_data.shape[2], 9))
    # RF_ss5_g2Fit = rerun_gridFit(RF_ss5_gFit, timeseries_data, stimulus, param_width, RF_ss5_g2Fit, indices)
    # tstamp_grid2fit = time.perf_counter()
    # print(f'Obtained grid2 estimates in {tstamp_grid2fit-tstamp_gridestim} seconds')

    # # Save the results
    # popeye_g2Fit = nib.nifti1.Nifti1Image(RF_ss5_g2Fit, affine=func_data.affine, header=func_data.header)
    # nib.save(popeye_g2Fit, os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit', 'RF_ss5_g2Fit_popeye.nii.gz'))
    
    # f1, axs = plt.subplots(2, 4, figsize=(20, 10))
    # axs = axs.flatten()
    # for i in range(8):
    #     ax = axs[i]
    #     ax.plot(trueFit_data[visual_rois, i].flatten(), RF_ss5_g2Fit[visual_rois, i].flatten(), 'o')
    #     ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    #     ax.set_title(f"Grid-fit: {['theta', 'rsquared', 'rho', 'sigma','n', 'x', 'y', 'beta'][i]}")
    #     ax.set_xlabel('mrVista')
    #     ax.set_ylabel('Popeye')
    # plt.savefig(os.path.join(p['fig_dir'], 'gridfit2_comparison.png'), dpi=300)
    # plt.close(f1)

    # ############################  FINAL FIT ################################
    print('Starting final fit ...')
    # RF_left_fFit = np.empty((leftDataOrig.shape[0], 9))
    RF_right_fFit = np.empty((rightDataOrig.shape[0], 9))
    # RF_left_fFit = get_final_estims(RF_left_gFit, param_width, leftData, stimulus, RF_left_fFit, indicesLeft)
    RF_right_fFit = get_final_estims(RF_right_gFit, param_width, rightData, stimulus, RF_right_fFit, indicesRight)
    tstamp_finalestim = time.perf_counter()
    print(f'Obtained final estimates in {tstamp_finalestim-tstamp_gridestim} seconds')
    print_time(tstamp_gridestim, tstamp_finalestim, 'Final fit')

    # Save the results
    # save2gifti(RF_left_fFit, fpath=os.path.join(p['fitEstimDir'], 'RF_ss5_fFit_popeye_left.func.gii'))
    save2gifti(RF_right_fFit, fpath=os.path.join(p['fitEstimDir'], 'RF_ss5_fFit_popeye_right.func.gii'))

    # f2, axs = plt.subplots(2, 4, figsize=(20, 10))
    # axs = axs.flatten()
    # for i in range(8):
    #     ax = axs[i]
    #     ax.hist(RF_left_fFit[indicesLeft, i].flatten(), bins=50, alpha=0.5, label='Left')
    #     # ax.hist(RF_right_fFit[indicesRight, i].flatten(), bins=50, alpha=0.5, label='Right')
    #     ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    #     ax.set_title(f"Final-fit: {['theta', 'rsquared', 'rho', 'sigma','n', 'x', 'y', 'beta'][i]}")
    #     ax.set_xlabel('mrVista') 
    #     ax.set_ylabel('Popeye')
    # plt.savefig(os.path.join(p['fig_dir'], 'finalfit_comparison.png'), dpi=300)
    # plt.close(f2)
    
    codeEndTime = time.perf_counter()
    print_time(codeStartTime, codeEndTime, 'Total time taken')


if __name__ == "__main__":
    main()
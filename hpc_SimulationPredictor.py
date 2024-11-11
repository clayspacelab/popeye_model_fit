## This is being set up to run popeye on HPC
import numpy as np
import ctypes, time, os
import pickle

# Import visualization stuff
import matplotlib.pyplot as plt
import nibabel as nib

# Import popeye stuff
import popeye.utilities_cclab as utils
from popeye.visual_stimulus import VisualStimulus

from itertools import product

# Load helper functions
from dataloader import *
from fit_utils import *
from fitutils_css import *
import ctypes

def main():
    # gridfit2
    # Initialize parameters
    print('Initializing parameters...')
    codeStartTime = time.perf_counter()
    params = {}
    params['subjID'] = 'JC'
    # Got these from Zhengang, and he got it from rsvp_params.txt
    params['viewingDistance'] = 83.5 #63 #83.5 # in cm
    params['screenWidth'] = 36.2 #35 #36.2 # in cm
    params['scaleFactor'] = 1
    params['resampleFactor'] = 1
    params['dtype'] = ctypes.c_int16

    p = set_paths(params)

    # Load stimulus
    bar, _ = load_stimuli(p)
    bar = bar[:, :, 0:201]
    # Mirror y axis (this is done because popeye flips the y axis)
    bar = np.flip(bar, axis=0)

    # copy_files(p, params)

    # Extract number of TRs
    method = 'ss5'
    func_data = nib.load(p['pRF_' + method])
    f_header = func_data.header
    params['tr_length'] = f_header['pixdim'][4]
    params['voxel_size'] = [f_header['pixdim'][i] for i in range(1, 4)]
    params['nTRs'] = func_data.shape[-1]

    # Load scan data
    print('Loading scan data...')
    simDataPath = os.path.join(p['pRF_data'], 'Simulation', 'simulatedVoxels.pkl')
    with open(simDataPath, 'rb') as f:
        scan_data = pickle.load(f)
    
    # load true fit data
    trueFitPath = os.path.join(p['pRF_data'], 'Simulation', 'simulatedParams.pkl')
    with open(trueFitPath, 'rb') as f:
        trueFitFile = pickle.load(f)
        trueFit_estims = np.asarray(trueFitFile['params_vox'])
        baseline_vox = trueFitFile['baseline_vox']

        # This is originally (x, y, sigma, n)
        # We need to convert it to (theta, r2, rho, sigma, n, x, y, beta, baseline)
        trueFit_data = np.empty((trueFit_estims.shape[0], 9))
        trueFit_data[:, 0] = np.mod(np.arctan2(trueFit_estims[:, 1], trueFit_estims[:, 0]), 2*np.pi)
        trueFit_data[:, 1] = 1
        trueFit_data[:, 2] = np.sqrt(trueFit_estims[:, 0]**2 + trueFit_estims[:, 1]**2)
        trueFit_data[:, 3] = trueFit_estims[:, 2]
        trueFit_data[:, 4] = trueFit_estims[:, 3]
        trueFit_data[:, 5] = trueFit_estims[:, 0]
        trueFit_data[:, 6] = trueFit_estims[:, 1]
        trueFit_data[:, 7] = np.zeros(trueFit_estims.shape[0])
        trueFit_data[:, 8] = baseline_vox

    # Select first N voxels
    nvox = 10
    scan_data = scan_data[:nvox, :]
    trueFit_data = trueFit_data[:nvox, :]
    scan_data_orig = scan_data.copy()
    scan_data = remove_trend(scan_data, method='all')
    # Plot 5 random voxels before and after detrending
    f, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(5):
        ax = axs[0, i]
        ax.plot(scan_data_orig[i])
        ax.set_title('Original')
        ax = axs[1, i]
        ax.plot(scan_data[i])
        ax.set_title('Detrended')
    plt.savefig(os.path.join(p['pRF_data'], 'Simulation/figures/detrended_voxels.png'), dpi=300)

    nvoxs = scan_data.shape[0]
    print(f"Running model-fit on {nvoxs} voxels")

    # print(f"Running model-fit on {len(np.argwhere(brainmask_data))} voxels")
    # scan_data_brainmask = scan_data.copy()
    # print(scan_data_brainmask.shape)
    # [xi, yi, zi] = np.nonzero(scan_data_brainmask)
    # indices = [(xi[i], yi[i], zi[i]) for i in range(len(xi))]
    # num_voxels = len(indices)
    timeseries_data = scan_data.copy()#scan_data_brainmask[xi, yi, zi, :]
    indices = [(0, 0, i) for i in range(nvoxs)]
    # print(f"Running model-fit on {num_voxels} voxels")

    # create stimulus object from popeye
    print('Creating stimulus object...')
    stimulus = VisualStimulus(bar.astype('int16'),
                            params['viewingDistance'],
                            params['screenWidth'],
                            params['scaleFactor'],
                            params['tr_length'],
                            params['dtype'],
    )

    # set search grids
    Ns = 50
    x_grid = np.concatenate((np.linspace(-stimulus.deg_x0.max(), stimulus.deg_x0.max(), Ns//2),
                        np.geomspace(-stimulus.deg_x0.max(), -2*stimulus.deg_x0.max(), Ns//4),
                            np.geomspace(stimulus.deg_x0.max(), 2*stimulus.deg_x0.max(), Ns//4)))
    y_grid = np.concatenate((np.linspace(-stimulus.deg_y0.max(), stimulus.deg_y0.max(), Ns//2),
                            np.geomspace(-stimulus.deg_y0.max(), -2*stimulus.deg_y0.max(), Ns//4),
                            np.geomspace(stimulus.deg_y0.max(), 2*stimulus.deg_y0.max(), Ns//4)))
    s_grid = np.concatenate((np.linspace(0.1, 5, 3*Ns//4), np.geomspace(5, stimulus.deg_x0.max(), Ns//4)))
    n_grid = np.asarray([0.25, 0.5, 0.75, 1])
    # n_grid = np.linspace(0.01, 1, Ns)
    grid_space_orig = list(product(x_grid, y_grid, s_grid, n_grid))
    grid_space = constraint_grids(grid_space_orig, stimulus)
    print(f'Number of grid points: {len(grid_space)}')

    param_width = [np.mean(np.diff(x_grid)), np.mean(np.diff(y_grid)), np.mean(np.diff(s_grid)), np.mean(np.diff(n_grid))]
    # param_width = np.asarray(round(param_width, 4))
    
    tstamp_start = time.perf_counter()
    
    if os.path.exists(p['gridfit_path']):
        print("Loading grid predictions from disk")
        grid_preds = np.load(p['gridfit_path'])
        # Print shape 
        print(grid_preds.shape)
    else:
        print("Grid predictions don't exist. Generating them")
        grid_preds = getGridPreds(grid_space, stimulus, p, timeseries_data)

    tstamp_gridpred = time.perf_counter()
    print_time(tstamp_start, tstamp_gridpred, 'Grid predictions')
    
    ############################  GRID FIT ################################
    print('Starting grid fit...')
    RF_ss5_gFit = np.empty((1, 1, timeseries_data.shape[0], 9))
    RF_ss5_gFit = get_grid_estims(grid_preds, grid_space, timeseries_data, RF_ss5_gFit, indices, use_gpu=False)
    tstamp_gridestim = time.perf_counter()
    print_time(tstamp_gridpred, tstamp_gridestim, 'Grid fit1')

    # Save the results
    params['subjID'] = 'Simulation'
    popeye_gFit = nib.nifti1.Nifti1Image(RF_ss5_gFit, affine=func_data.affine, header=func_data.header)
    if not os.path.exists(os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit')):
        os.makedirs(os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit'))
    nib.save(popeye_gFit, os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit', 'RF_ss5_gFit_popeye.nii.gz'))

    f0, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for i in range(8):
        ax = axs[i]
        ax.plot(trueFit_data[:,i].flatten(), RF_ss5_gFit[:, :, :, i].flatten(), 'o')
        ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
        ax.set_title(f"Grid-fit: {['theta', 'rsquared', 'rho', 'sigma','n', 'x', 'y', 'beta'][i]}")
        ax.set_xlabel('GroundTruth')
        ax.set_ylabel('Popeye')
    plt.savefig(os.path.join(p['pRF_data'], 'Simulation/figures/gridfit_comparison.png'), dpi=300)
    plt.close(f0)

    ############################  GRID FIT2 ################################
    # RF_ss5_g2Fit = np.empty((1, 1, timeseries_data.shape[0], 9))
    # RF_ss5_g2Fit = rerun_gridFit_parallel(RF_ss5_gFit, timeseries_data, stimulus, param_width, RF_ss5_g2Fit, indices, use_gpu=False)
    # tstamp_grid2fit = time.perf_counter()
    # print(f'Obtained grid2 estimates in {tstamp_grid2fit-tstamp_gridestim} seconds')

    # # Save the results
    # popeye_g2Fit = nib.nifti1.Nifti1Image(RF_ss5_g2Fit, affine=func_data.affine, header=func_data.header)
    # nib.save(popeye_g2Fit, os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit', 'RF_ss5_g2Fit_popeye.nii.gz'))
    
    # f1, axs = plt.subplots(2, 4, figsize=(20, 10))
    # axs = axs.flatten()
    # for i in range(8):
    #     ax = axs[i]
    #     ax.plot(trueFit_data[:, i].flatten(), RF_ss5_g2Fit[:, :, :, i].flatten(), 'o')
    #     ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    #     ax.set_title(f"Grid-fit: {['theta', 'rsquared', 'rho', 'sigma','n', 'x', 'y', 'beta'][i]}")
    #     ax.set_xlabel('GroundTruth')
    #     ax.set_ylabel('Popeye')
    # plt.savefig(os.path.join(p['pRF_data'], 'Simulation/figures/gridfit2_comparison.png'), dpi=300)
    # plt.close(f1)

    ############################  FINAL FIT ################################
    RF_ss5_fFit = np.empty((1, 1, timeseries_data.shape[0], 9))
    RF_ss5_fFit = get_final_estims(RF_ss5_gFit, param_width, timeseries_data, stimulus, RF_ss5_fFit, indices, use_gpu=False)
    tstamp_finalestim = time.perf_counter()
    # print(f'Obtained final estimates in {tstamp_finalestim-tstamp_gridestim} seconds')
    print_time(tstamp_gridestim, tstamp_finalestim, 'Final fit')

    # Save the results
    popeye_fFit = nib.nifti1.Nifti1Image(RF_ss5_fFit, affine=func_data.affine, header=func_data.header)
    nib.save(popeye_fFit, os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit', 'RF_ss5_fFit_popeye.nii.gz'))

    f2, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for i in range(8):
        ax = axs[i]
        ax.plot(trueFit_data[:, i].flatten(), RF_ss5_fFit[:, :, :, i].flatten(), 'o')
        ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
        ax.set_title(f"Final-fit: {['theta', 'rsquared', 'rho', 'sigma','n', 'x', 'y', 'beta'][i]}")
        ax.set_xlabel('GroundTruth') 
        ax.set_ylabel('Popeye')
    plt.savefig(os.path.join(p['pRF_data'], 'Simulation/figures/finalfit_comparison.png'), dpi=300)
    plt.close(f2)
    
    codeEndTime = time.perf_counter()
    print_time(codeStartTime, codeEndTime, 'Total time taken')

if __name__ == "__main__":
    main()
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
from dataloader import *
from fit_utils import *
from fitutils_css import *
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

    p = set_paths(params)

    # Load stimulus
    bar, _ = load_stimuli(p)
    bar = bar[:, :, 0:201]
    # Mirror y axis (this is done because popeye flips the y axis)
    bar = np.flip(bar, axis=0)

    copy_files(p, params)

    # Extract number of TRs
    method = 'ss5'
    func_data = nib.load(p['pRF_' + method])
    f_header = func_data.header
    params['tr_length'] = f_header['pixdim'][4]
    params['voxel_size'] = [f_header['pixdim'][i] for i in range(1, 4)]
    params['nTRs'] = func_data.shape[-1]

    # load scan data
    print('Loading scan data ...')
    scan_data = func_data.get_fdata()
    scan_data = remove_trend(scan_data, method='all')

    brainmask_data = nib.load(p['pRF_brainmask']).get_fdata() != 0
    # Resample brainmask if first 2 dimensions are twice the third dimension
    if brainmask_data.shape[0] == 2*brainmask_data.shape[2]:
        brainmask_data = brainmask_data[::2, ::2, :]

    nan_voxs = np.isnan(scan_data).any(axis=-1)
    brainmask_data = brainmask_data * ~nan_voxs

    # Check number of voxels in brainmask
    print(f"Number of voxels in brainmask: {len(np.argwhere(brainmask_data))}")

    nvoxs = 1000
    print(f"Running model-fit on {nvoxs} voxels")
    # Select 100 random voxels from brainmask
    voxels = np.argwhere(brainmask_data)
    np.random.shuffle(voxels)
    voxels = voxels[:nvoxs]
    brainmask_data = np.zeros_like(brainmask_data)
    for voxel in voxels:
        brainmask_data[voxel[0], voxel[1], voxel[2]] = 1

    # print(f"Running model-fit on {len(np.argwhere(brainmask_data))} voxels")
    scan_data_brainmask = scan_data.copy()
    scan_data_brainmask[~brainmask_data] = 0
    [xi, yi, zi] = np.nonzero(brainmask_data)
    indices = [(xi[i], yi[i], zi[i]) for i in range(len(xi))]
    num_voxels = len(indices)
    timeseries_data = scan_data_brainmask[xi, yi, zi, :]
    # print(f"Running model-fit on {num_voxels} voxels")

    # create stimulus object from popeye
    print('Creating stimulus object ...')
    stimulus = VisualStimulus(bar.astype('int16'),
                            params['viewingDistance'],
                            params['screenWidth'],
                            params['scaleFactor'],
                            params['tr_length'],
                            params['dtype'],
    )

    # Testing only on visual ROIs
    # Load visual ROIs
    # lh_v1 = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'roi_mdd', 'lh.V1.nii.gz')).get_fdata()
    # lh_v2d = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'roi_mdd', 'lh.V2d.nii.gz')).get_fdata()
    # lh_v3d = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'roi_mdd', 'lh.V3d.nii.gz')).get_fdata()
    # lh_v3ab = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'roi_mdd', 'lh.V3AB.nii.gz')).get_fdata()
    # rh_v1 = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'roi_mdd', 'rh.V1.nii.gz')).get_fdata()
    # rh_v2d = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'roi_mdd', 'rh.V2d.nii.gz')).get_fdata()
    # rh_v3d = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'roi_mdd', 'rh.V3d.nii.gz')).get_fdata()
    # rh_v3ab = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'roi_mdd', 'rh.V3AB.nii.gz')).get_fdata()
    # # Combine all ROIs using boolean OR
    # visual_rois = lh_v1 + lh_v2d + lh_v3d + lh_v3ab + rh_v1 + rh_v2d + rh_v3d + rh_v3ab
    # visual_rois = visual_rois > 0

    # nan_voxs = np.isnan(scan_data).any(axis=-1)
    # visual_rois = visual_rois * ~nan_voxs
    # # print(len(np.argwhere(visual_rois)))
    # print(f"Running model-fit on {len(np.argwhere(visual_rois))} voxels")

    trueFit_data = nib.load(os.path.join(p['pRF_data'], params['subjID'], 'mrVistaFit/RF_ss5-fFit.nii.gz')).get_fdata()

    # nvoxs = 100
    # print(f"Running model-fit on {nvoxs} voxels")
    # # Select 100 random voxels from visual ROIs
    # voxels = np.argwhere(visual_rois)
    # np.random.shuffle(voxels)
    # voxels = voxels[:nvoxs]
    # visual_rois = np.zeros_like(visual_rois)
    # for voxel in voxels:
    #     visual_rois[voxel[0], voxel[1], voxel[2]] = 1

    # Create scan data just for visual ROIs
    # scan_data_visual = scan_data.copy()
    # scan_data_visual[~visual_rois] = 0

    # [xi, yi, zi] = np.nonzero(visual_rois)
    # indices = [(xi[i], yi[i], zi[i]) for i in range(len(xi))]
    # num_voxels = len(indices)
    # timeseries_data = scan_data_visual[xi, yi, zi, :]

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
    grid_space_orig = list(product(x_grid, y_grid, s_grid, n_grid))
    grid_space = constraint_grids(grid_space_orig, stimulus)
    print(f'Number of grid points: {len(grid_space)}')

    param_width = [np.mean(np.diff(x_grid)), np.mean(np.diff(y_grid)), np.mean(np.diff(s_grid)), np.mean(np.diff(n_grid))]
    # param_width = np.asarray(round(param_width, 4))
    
    tstamp_start = time.perf_counter()
    
    if os.path.exists(p['gridfit_path']):
        print("Loading grid predictions from disk")
        grid_preds = np.load(p['gridfit_path'])
    else:
        print("Grid predictions don't exist. Generating them")
        grid_preds = getGridPreds(grid_space, stimulus, p, timeseries_data)

    tstamp_gridpred = time.perf_counter()
    # print(f'Obtained grid predictions in {tstamp_gridpred-tstamp_start} seconds')
    print_time(tstamp_start, tstamp_gridpred, 'Grid predictions')
    
    ############################  GRID FIT ################################
    print('Starting grid fit ...')
    RF_ss5_gFit = np.empty((scan_data.shape[0], scan_data.shape[1], scan_data.shape[2], 9))
    RF_ss5_gFit = get_grid_estims(grid_preds, grid_space, timeseries_data, RF_ss5_gFit, indices)
    tstamp_gridestim = time.perf_counter()
    # print(f'Obtained grid estimates in {tstamp_gridestim-tstamp_gridpred} seconds')
    print_time(tstamp_gridpred, tstamp_gridestim, 'Grid fit1')

    # Save the results
    popeye_gFit = nib.nifti1.Nifti1Image(RF_ss5_gFit, affine=func_data.affine, header=func_data.header)
    if not os.path.exists(os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit')):
        os.makedirs(os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit'))
    nib.save(popeye_gFit, os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit', 'RF_ss5_gFit_popeye.nii.gz'))

    f0, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    for i in range(8):
        ax = axs[i]
        ax.plot(trueFit_data[brainmask_data, i].flatten(), RF_ss5_gFit[brainmask_data, i].flatten(), 'o')
        ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
        ax.set_title(f"Grid-fit: {['theta', 'rsquared', 'rho', 'sigma','n', 'x', 'y', 'beta'][i]}")
        ax.set_xlabel('mrVista')
        ax.set_ylabel('Popeye')
    plt.savefig(os.path.join(p['fig_dir'], 'gridfit_comparison.png'), dpi=300)
    plt.close(f0)

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
    # RF_ss5_fFit = np.empty((scan_data.shape[0], scan_data.shape[1], scan_data.shape[2], 9))
    # RF_ss5_fFit = get_final_estims(RF_ss5_gFit, param_width, timeseries_data, stimulus, RF_ss5_fFit, indices)
    # tstamp_finalestim = time.perf_counter()
    # # print(f'Obtained final estimates in {tstamp_finalestim-tstamp_gridestim} seconds')
    # print_time(tstamp_gridestim, tstamp_finalestim, 'Final fit')

    # # Save the results
    # popeye_fFit = nib.nifti1.Nifti1Image(RF_ss5_fFit, affine=func_data.affine, header=func_data.header)
    # nib.save(popeye_fFit, os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit', 'RF_ss5_fFit_popeye.nii.gz'))

    # f2, axs = plt.subplots(2, 4, figsize=(20, 10))
    # axs = axs.flatten()
    # for i in range(8):
    #     ax = axs[i]
    #     ax.plot(trueFit_data[visual_rois, i].flatten(), RF_ss5_fFit[visual_rois, i].flatten(), 'o')
    #     ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    #     ax.set_title(f"Final-fit: {['theta', 'rsquared', 'rho', 'sigma','n', 'x', 'y', 'beta'][i]}")
    #     ax.set_xlabel('mrVista') 
    #     ax.set_ylabel('Popeye')
    # plt.savefig(os.path.join(p['fig_dir'], 'finalfit_comparison.png'), dpi=300)
    # plt.close(f2)
    
    codeEndTime = time.perf_counter()
    print_time(codeStartTime, codeEndTime, 'Total time taken')
    # Save the results
    # anat_data = nib.load(p['pRF_anat']) #.get_fdata()
    # # popeye_fFit = nib.Nifti1Image(RF_ss5_fFit, affine=anat_data.affine, header=anat_data.header)
    # popeye_fFit = nib.nifti1.Nifti1Image(RF_ss5_fFit, affine=func_data.affine, header=func_data.header)
    # # popeye_gFit = nib.Nifti1Image(RF_ss5_gFit, affine=anat_data.affine, header=anat_data.header)
    # popeye_gFit = nib.nifti1.Nifti1Image(RF_ss5_gFit, affine=func_data.affine, header=func_data.header)
    # if not os.path.exists(os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit')):
    #     os.makedirs(os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit'))
    # nib.save(popeye_fFit, os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit', 'RF_ss5_fFit_popeye.nii.gz'))
    # nib.save(popeye_gFit, os.path.join(p['pRF_data'], params['subjID'], 'popeyeFit', 'RF_ss5_gFit_popeye.nii.gz'))

    # # Set search bounds
    # x_bounds = (-15.0, 15.0)
    # y_bounds = (-15.0, 15.0)
    # s_bounds = (1/css_model.stimulus.ppd0, 5.25)
    # n_bounds = (0.2, 1)
    # b_bounds = (1e-8, None)
    # m_bounds = (None, None)
    # bounds = (x_bounds, y_bounds, s_bounds, n_bounds, b_bounds, m_bounds)

    # verbose = 0
    # auto_fit = 1

    # # Create a result holder
    # RF_ss5_gFit = np.empty((scan_data_visual.shape[0], scan_data_visual.shape[1], scan_data_visual.shape[2], 9))
    # RF_ss5_fFit = np.empty((scan_data_visual.shape[0], scan_data_visual.shape[1], scan_data_visual.shape[2], 9))
    # vx_indices = np.argwhere(visual_rois)


if __name__ == "__main__":
    main()
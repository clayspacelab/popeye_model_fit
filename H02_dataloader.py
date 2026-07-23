"""
H02_dataloader.py — Unified data loading for volumetric and surface fMRI data.

Consolidates the previously separate dataloader.py (volumetric) and
helpersSurface.py (surface) into a single module.

Key functions:
    load_stimuli()          — Load bar stimulus masks (.mat)
    load_volumetric_data()  — Load NIfTI volumetric data
    load_surface_data()     — Load GIFTI surface data, average runs
    save2gifti()            — Save data as GIFTI
    save2nifti()            — Save data as NIfTI
    copy_files()            — Copy original data to pRF directory
"""

import os
import numpy as np
import shutil
from scipy.io import loadmat
import nibabel as nib


# ---------------------------------------------------------------------------
# Stimulus loading (shared across formats)
# ---------------------------------------------------------------------------
def load_stimuli(p):
    """
    Load the bar stimulus masks and params files.

    Parameters
    ----------
    p : dict
        Path dictionary from H01_config.set_paths().

    Returns
    -------
    bar : ndarray
        Stimulus image array (x, y, time).
    params : dict
        Stimulus parameters from .mat file.
    """
    bar = loadmat(
        os.path.join(p['stimuli_path'], 'bar_stimulus_masks_1300ms_images.mat')
    )['images']
    params = loadmat(
        os.path.join(p['stimuli_path'], 'bar_stimulus_masks_1300ms_params.mat')
    )
    return bar, params


# ---------------------------------------------------------------------------
# Volumetric data loading
# ---------------------------------------------------------------------------
def load_volumetric_data(p, method='ss5'):
    """
    Load volumetric (NIfTI) fMRI data and extract metadata.

    Parameters
    ----------
    p : dict
        Path dictionary from H01_config.set_paths().
    method : str
        Data variant to load ('ss5', 'func', 'surf'). Default 'ss5'.

    Returns
    -------
    scan_data : ndarray
        4D fMRI data array (x, y, z, time).
    func_img : nib.Nifti1Image
        NIfTI image object (for affine/header).
    metadata : dict
        Extracted metadata: tr_length, voxel_size, nTRs.
    """
    fpath = p['pRF_' + method]
    func_img = nib.load(fpath)
    scan_data = func_img.get_fdata()
    header = func_img.header

    metadata = {
        'tr_length': float(header['pixdim'][4]),
        'voxel_size': [float(header['pixdim'][i]) for i in range(1, 4)],
        'nTRs': scan_data.shape[-1],
    }

    return scan_data, func_img, metadata


def extract_brainmask_voxels(scan_data, brainmask_data=None):
    """
    Extract non-zero voxel timeseries and their 3D indices from volumetric data.

    Parameters
    ----------
    scan_data : ndarray
        4D fMRI data (x, y, z, time).
    brainmask_data : ndarray or None
        Optional 3D binary brainmask. If None, uses non-zero voxels from scan_data.

    Returns
    -------
    timeseries_data : ndarray
        2D array (n_voxels, n_timepoints).
    indices : list of tuple
        List of (x, y, z) index tuples for each voxel.
    """
    if brainmask_data is not None:
        xi, yi, zi = np.nonzero(brainmask_data)
    else:
        # Use voxels with non-zero mean signal
        mean_signal = np.mean(scan_data, axis=-1)
        xi, yi, zi = np.nonzero(mean_signal)

    indices = [(xi[i], yi[i], zi[i]) for i in range(len(xi))]
    timeseries_data = scan_data[xi, yi, zi, :]

    return timeseries_data, indices


# ---------------------------------------------------------------------------
# Surface data loading
# ---------------------------------------------------------------------------
def load_surface_data(p, funcFiles):
    """
    Load surface (GIFTI) fMRI data, averaging across runs.

    Parameters
    ----------
    p : dict
        Path dictionary from H01_config.set_paths().
    funcFiles : list of str
        List of functional GIFTI filenames.

    Returns
    -------
    leftData : ndarray
        Left hemisphere data (n_vertices, n_timepoints).
    rightData : ndarray
        Right hemisphere data (n_vertices, n_timepoints).
    tr_length : float
        TR duration in seconds.
    nTRs : int
        Number of timepoints.
    """
    avgLeftfName = funcFiles[0].replace('run-01_hemi-L_space', 'hemi-L_avg')
    avgRightfName = funcFiles[0].replace('run-01_hemi-R_space', 'hemi-R_avg')

    left_avg_path = os.path.join(p['pRF_avgRoot'], avgLeftfName)
    right_avg_path = os.path.join(p['pRF_avgRoot'], avgRightfName)

    if os.path.exists(left_avg_path) and os.path.exists(right_avg_path):
        print("Loading average data from disk")
        imgLeft = nib.load(left_avg_path)
        leftData = np.array([x.data for x in imgLeft.darrays]).T
        imgRight = nib.load(right_avg_path)
        rightData = np.array([x.data for x in imgRight.darrays]).T
    else:
        leftFiles = [f for f in funcFiles if 'L_space' in f]
        rightFiles = [f for f in funcFiles if 'R_space' in f]
        nRuns = len(leftFiles)
        print(f'Number of runs: {nRuns}')

        for run in range(nRuns):
            leftFname = os.path.join(p['pRF_root'], leftFiles[run])
            rightFname = os.path.join(p['pRF_root'], rightFiles[run])

            imgLeft = nib.load(leftFname)
            tempLeft = np.array([x.data for x in imgLeft.darrays])
            tempLeft = np.expand_dims(tempLeft, axis=-1)

            imgRight = nib.load(rightFname)
            tempRight = np.array([x.data for x in imgRight.darrays])
            tempRight = np.expand_dims(tempRight, axis=-1)

            if run == 0:
                leftData = tempLeft
                rightData = tempRight
            else:
                leftData = np.concatenate((leftData, tempLeft), axis=-1)
                rightData = np.concatenate((rightData, tempRight), axis=-1)

        # Average the runs
        leftData = np.mean(leftData, axis=-1).T
        rightData = np.mean(rightData, axis=-1).T

        # Save the average data
        save2gifti(leftData, left_avg_path, 'left')
        save2gifti(rightData, right_avg_path, 'right')

    tr_length = 1.3  # seconds
    nTRs = leftData.shape[1]

    return leftData, rightData, tr_length, nTRs


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------
def save2gifti(data, fpath, hemisphere):
    """
    Save a 2D data array as a GIFTI .func.gii file.

    Parameters
    ----------
    data : ndarray
        2D array (n_vertices, n_params).
    fpath : str
        Output file path.
    hemisphere : str
        'left' or 'right'.
    """
    data = data.astype(np.float32)
    if hemisphere == 'left':
        anat_struct = 'CortexLeft'
    elif hemisphere == 'right':
        anat_struct = 'CortexRight'
    else:
        raise ValueError(f"hemisphere must be 'left' or 'right', got '{hemisphere}'")

    giiMeta = nib.gifti.GiftiMetaData({
        'AnatomicalStructurePrimary': anat_struct,
        'PaletteNormalizationMode': 'NORMALIZATION_SELECTED_MAP_DATA',
        'TimeStep': '1.3',
    })
    img = nib.gifti.GiftiImage(meta=giiMeta)
    for i in range(data.shape[1]):
        thisParam = nib.gifti.GiftiDataArray(data=data[:, i])
        img.add_gifti_data_array(thisParam)
    nib.save(img, fpath)


def save2nifti(data, fpath, affine, header):
    """
    Save data as a NIfTI .nii.gz file.

    Parameters
    ----------
    data : ndarray
        Data array to save.
    fpath : str
        Output file path.
    affine : ndarray
        4x4 affine matrix.
    header : nib.Nifti1Header
        NIfTI header.
    """
    img = nib.nifti1.Nifti1Image(data, affine=affine, header=header)
    nib.save(img, fpath)


# ---------------------------------------------------------------------------
# File copy utility (volumetric only)
# ---------------------------------------------------------------------------
def copy_files(p, subjID):
    """
    Copy original data files to the pRF working directory (volumetric only).

    Parameters
    ----------
    p : dict
        Path dictionary (must contain 'orig_*' and 'pRF_*' keys).
    subjID : str
        Subject identifier.
    """
    subj_dir = os.path.join(p['pRF_data'], subjID)
    if not os.path.exists(subj_dir):
        os.makedirs(subj_dir)
        for key_suffix in ('brainmask', 'func', 'ss5', 'surf', 'anat'):
            orig_key = f'orig_{key_suffix}'
            prf_key = f'pRF_{key_suffix}'
            if orig_key in p and prf_key in p:
                shutil.copy2(p[orig_key], p[prf_key])
        print(f"Copied data for {subjID}")
    else:
        print('Subject folder already exists')

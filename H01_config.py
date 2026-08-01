"""
H01_config.py — Central configuration for the pRF fitting pipeline.

Consolidates experiment parameters, host detection, and path generation
from the previously separate dataloader.py and helpersSurface.py modules.

Supports both volumetric (NIfTI, default) and surface (GIFTI) data formats.
"""

import os
import socket
import ctypes

import numpy as np

# ---------------------------------------------------------------------------
# Default experiment parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    'viewingDistance': 83.5,   # cm (from Zhengang / rsvp_params.txt)
    'screenWidth': 36.2,      # cm
    'scaleFactor': 1,
    'resampleFactor': 1,
    'dtype': ctypes.c_int16,
    'tr_length': 1.3,         # seconds
}

# Grid search defaults
GRID_DEFAULTS = {
    'Ns': 50,                 # grid density
    'n_grid_values': [0.25, 0.5, 0.75, 1.0],  # coarse CSS exponent grid (subject pipeline)
    # Finer 10-value CSS exponent grid used by the simulation scripts (S02/S03),
    # since the exponent is the parameter most sensitive to grid resolution.
    'n_grid_values_fine': np.round(np.linspace(0.25, 1.0, 10), 4).tolist(),
}

def get_gridfit_path(p, Ns=None, n_res=None):
    """Return the cached grid-prediction path.

    Caches live in a dedicated ``Stimuli/gridestims/`` folder (created on demand).
    When ``n_res`` (the CSS-exponent grid resolution) is given, the filename also
    encodes it — ``gridfit_Ns{Ns}_n{n_res}.npy`` — so caches built with different
    n-grid densities never collide and are shared across scripts (e.g. S02 reuses
    the file S03 generated). Without ``n_res`` the legacy ``gridfit_{Ns}.npy`` name
    is used.
    """
    if Ns is None:
        Ns = GRID_DEFAULTS['Ns']
    gridestims_dir = os.path.join(p['stimuli_path'], 'gridestims')
    os.makedirs(gridestims_dir, exist_ok=True)
    if n_res is not None:
        return os.path.join(gridestims_dir, f'gridfit_Ns{Ns}_n{n_res}.npy')
    return os.path.join(gridestims_dir, f'gridfit_{Ns}.npy')

# CSS model output field names (9-element tuple per voxel/vertex)
CSS_FIELD_NAMES = (
    'theta',     # polar angle
    'r2',        # variance explained (R²)
    'rho',       # eccentricity
    'sigma',     # RF size
    'n',         # CSS exponent
    'x',         # x-position
    'y',         # y-position
    'beta',      # amplitude (slope)
    'baseline',  # baseline (intercept)
)


# ---------------------------------------------------------------------------
# Host detection
# ---------------------------------------------------------------------------
def detect_host():
    """
    Detect the compute environment based on hostname.

    Returns
    -------
    str
        One of: 'lab_local', 'vader', 'local_mac', 'unknown'
    """
    hostname = socket.gethostname()

    if hostname in ('syndrome', 'zod.psych.nyu.edu', 'zod'):
        return 'lab_local'
    elif 'vader' in hostname:
        return 'vader'
    elif 'Mrugank' in hostname or 'mrugank' in hostname:
        return 'local_mac'
    else:
        return 'unknown'


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------
def set_paths(subjID, data_format='volumetric'):
    """
    Build all file paths for a given subject and data format.

    Parameters
    ----------
    subjID : str
        Subject identifier (e.g., 'MAM0606').
    data_format : str
        'volumetric' (NIfTI, default) or 'surface' (GIFTI / fMRIPrep).

    Returns
    -------
    p : dict
        Dictionary of all relevant file paths.
    funcFiles : list or None
        List of functional GIFTI filenames (surface mode only). None for volumetric.
    """
    host = detect_host()
    p = {}
    p['hostname'] = socket.gethostname()
    p['host_type'] = host
    funcFiles = None

    # --- Root data directories ---
    if host == 'lab_local':
        p['pRF_data'] = '/d/DATD/datd/popeye_pRF/'
        p['orig_data'] = '/d/DATD/datd/pRF_orig/'
    elif host == 'vader':
        p['pRF_data'] = '/clayspace/datd/popeye_pRF/'
    elif host == 'local_mac':
        p['pRF_data'] = '/Users/mrugankdake/Documents/Clayspace/MRI/popeye_pRF/'
        p['orig_data'] = '/Users/mrugankdake/Documents/Clayspace/MRI/pRF_orig/'
    else:
        raise ValueError(f"Unknown host '{socket.gethostname()}'. "
                         "Please add path configuration for this host in H01_config.py.")

    # --- Stimuli paths (shared) ---
    p['stimuli_path'] = os.path.join(p['pRF_data'], 'Stimuli')
    p['gridfit_path'] = get_gridfit_path(p)

    # --- Data-format-specific paths ---
    if data_format == 'volumetric':
        p = _set_volumetric_paths(p, subjID, host)
    elif data_format == 'surface':
        p, funcFiles = _set_surface_paths(p, subjID)
    else:
        raise ValueError(f"Unknown data_format '{data_format}'. Use 'volumetric' or 'surface'.")

    # --- Output directories (shared) ---
    subj_dir = os.path.join(p['pRF_data'], subjID) if data_format == 'volumetric' \
               else os.path.join(p['pRF_data'], 'sub-' + subjID)
    os.makedirs(subj_dir, exist_ok=True)

    p['popeyeFitDir'] = os.path.join(subj_dir, 'popeyeFit')
    os.makedirs(p['popeyeFitDir'], exist_ok=True)

    p['fig_dir'] = os.path.join(p['popeyeFitDir'], 'figs')
    os.makedirs(p['fig_dir'], exist_ok=True)

    p['fitEstimDir'] = os.path.join(p['popeyeFitDir'], 'fitEstimates')
    os.makedirs(p['fitEstimDir'], exist_ok=True)

    return p, funcFiles


def _set_volumetric_paths(p, subjID, host):
    """Set paths for volumetric (NIfTI) data."""
    # Original data paths (for copying, if available)
    if 'orig_data' in p:
        p['orig_brainmask'] = os.path.join(p['orig_data'], subjID,
                                           'surfanat_brainmask_hires.nii.gz')
        p['orig_func'] = os.path.join(p['orig_data'], subjID, 'RF1',
                                      subjID + '_RF1_vista', 'bar_seq_1_func.nii.gz')
        p['orig_ss5'] = os.path.join(p['orig_data'], subjID, 'RF1',
                                     subjID + '_RF1_vista', 'bar_seq_1_ss5.nii.gz')
        p['orig_surf'] = os.path.join(p['orig_data'], subjID, 'RF1',
                                      subjID + '_RF1_vista', 'bar_seq_1_surf.nii.gz')
        p['orig_anat'] = os.path.join(p['orig_data'], subjID, 'anat_T1_brain.nii')

    # pRF data paths
    p['pRF_brainmask'] = os.path.join(p['pRF_data'], subjID,
                                      'surfanat_brainmask_hires.nii.gz')
    p['pRF_func'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_func.nii.gz')
    p['pRF_ss5'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_ss5.nii.gz')
    p['pRF_surf'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_surf.nii.gz')
    p['pRF_anat'] = os.path.join(p['pRF_data'], subjID, 'anat_T1_brain.nii')

    return p


def _set_surface_paths(p, subjID):
    """Set paths for surface (GIFTI / fMRIPrep) data."""
    p['pRF_root'] = os.path.join(p['pRF_data'], 'sub-' + subjID,
                                 'ses-pRF', 'func')
    funcFiles = [f for f in os.listdir(p['pRF_root'])
                 if f.endswith('fsnative_bold.func.gii')]

    p['pRF_avgRoot'] = os.path.join(p['pRF_data'], 'sub-' + subjID,
                                    'ses-pRF', 'func_avg')
    os.makedirs(p['pRF_avgRoot'], exist_ok=True)

    return p, funcFiles

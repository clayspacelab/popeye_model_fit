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
"""

import os
import glob
import hashlib
import warnings
import numpy as np
from scipy.io import loadmat
import nibabel as nib


# ---------------------------------------------------------------------------
# Path generation
# Note: long-term it would make sense to make this flexibly update p dict based on a principled mapping
# between command-line inputs and config options, but leaving as is for now.
# ---------------------------------------------------------------------------
def set_paths(data, data_format, stimulus, defaults=None):
    """
    Build all file paths for a given subject and data format.

    Parameters
    ----------
    data : str
        Base directory/filename for the subject's data.
    data_format : str
        'volumetric' (NIfTI, default) or 'surface' (GIFTI / fMRIPrep).
    stimulus : str
        Path to the stimulus data file.
    defaults : dict
        Default parameters (e.g., from config.py). Used for various other settings.

    Returns
    -------
    p : dict
        Dictionary of all relevant file paths.
    """

    p = defaults if defaults is not None else {}
    p['pRF_dir'] = os.path.dirname(data)

    if data_format == 'volume':        
        if not os.path.isfile(data) or not data.endswith(('.nii.gz','.nii')):
            raise ValueError(f"Volumetric data path '{data}' must be a NIfTI (.nii or .nii.gz) file.")
        p['pRF_func'] = data

    elif data_format == 'surface':
        funcFiles = glob.glob(os.path.join(data, '*.func.gii'))
        if len(funcFiles) == 0 or len(funcFiles) > 2: # this is a weak check, but should be sufficient until data loaded
            raise ValueError(f"Surface data path '{data}' must be a prefix for a pair of GIFTI (func.gii) files if fitting two hemispheres, or a single GIFTI file.")
        p['pRF_func'] = funcFiles

    else:
        raise ValueError(f"Unknown data_format '{data_format}'. Use 'volume' or 'surface'.")

    #set stimulus path
    if not os.path.isfile(stimulus):
            raise ValueError(f"Stimulus path '{stimulus}' does not exist.")
    p['stimuli_path'] = stimulus

    # --- Output directories  ---
    p['popeyeFitDir'] = os.path.join(p['pRF_dir'], 'popeyeFit')
    os.makedirs(p['popeyeFitDir'], exist_ok=True)

    p['fig_dir'] = os.path.join(p['popeyeFitDir'], 'figs')
    os.makedirs(p['fig_dir'], exist_ok=True)

    p['fitEstimDir'] = os.path.join(p['popeyeFitDir'], 'fitEstimates')
    os.makedirs(p['fitEstimDir'], exist_ok=True)

    return p

# ---------------------------------------------------------------------------
# Stimulus loading (shared across formats)
# ---------------------------------------------------------------------------
def load_stimuli(p,flip_y=True):
    """
    Load the bar stimulus.

    Parameters
    ----------
    p : dict
        Path dictionary from set_paths().
    flip_y : bool (default True)
        If True, flip the y-axis of the stimulus to match the coordinate system
        used in the internal model fit.
    
    Returns
    -------
    bar : ndarray
        Stimulus image array (x, y, time).
    params : dict
        Stimulus parameters from .mat file.
    """

    stim_path = p['stimuli_path']
    ext = os.path.splitext(stim_path)[1].lower()

    if ext == '.mat':
        # Legacy support for (basic) Vista-style StimFromScan files.
        bar = loadmat(stim_path)['images']
    elif ext in ['.npy', '.npz']:
        bar = np.load(stim_path)
    else:
        raise ValueError(
            f"Unsupported stimulus file format '{ext}' for '{stim_path}'. "
            "Expected a MATLAB (.mat) file or a numpy (.npy/.npz) file."
        )

    if flip_y:
        bar = np.flip(bar, axis=0)  # Flip y-axis to match internal coordinate system

    return bar


# ---------------------------------------------------------------------------
# Grid-prediction caching
# ---------------------------------------------------------------------------
def _hash_grid_inputs(grid_space, stim_params, hrf):
    """Short content hash of the grid-prediction inputs, for cache invalidation."""
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(grid_space).tobytes())
    hasher.update(np.ascontiguousarray(stim_params.stim_arr).tobytes())
    hasher.update(np.ascontiguousarray(stim_params.deg_x).tobytes())
    hasher.update(np.ascontiguousarray(stim_params.deg_y).tobytes())
    hasher.update(str(stim_params.run_length).encode())
    hasher.update(np.ascontiguousarray(hrf).tobytes())
    return hasher.hexdigest()[:10]  # short prefix; plenty for local collision avoidance


def get_gridfit_path(p, grid_space, stim_params, hrf, Ns, n_res):
    """
    Return the cached grid-prediction path.

    The filename keeps ``Ns`` (and ``n_res``, if given) for human readability,
    but also encodes a short content hash of the actual ``grid_space``,
    stimulus, and ``hrf`` used to generate the predictions. Any change to
    grid construction (xy_scale, s/n bounds, stimulus geometry, TR/hrf, etc.)
    changes the hash and so automatically invalidates stale caches, instead
    of silently reusing a cache that only matches on Ns.

    Parameters
    ----------
    p : dict
        Path dictionary from set_paths(); must contain 'stimuli_path'.
    grid_space : array-like
        Constrained grid points, as used to generate the predictions.
    stim_params : namedtuple
        Stimulus params (stim_arr, deg_x, deg_y, run_length), e.g. stimulus.params.
    hrf : ndarray
        Hemodynamic response function used for the predictions.
    Ns : int or None
        Grid density parameter, for the readable part of the filename.
    n_res : int or None
        CSS-exponent grid resolution, included in the filename if given.

    Returns
    -------
    str
        Path to the (possibly not-yet-existing) cached grid-predictions file.
    """
    gridestims_dir = os.path.join(os.path.dirname(p['stimuli_path']), 'gridestims')
    os.makedirs(gridestims_dir, exist_ok=True)

    digest = _hash_grid_inputs(grid_space, stim_params, hrf)
    ns_part = f'Ns{Ns}'
    n_res_part = f'n{n_res}'
    return os.path.join(gridestims_dir, f'gridfit_{ns_part}_{n_res_part}_{digest}.npy')


def _tr_to_sec(tr):
    """Convert TR length to seconds if it's in milliseconds."""
    if tr > 100:  # likely in ms
        tr /= 1000.0
    return tr

# ---------------------------------------------------------------------------
# Volumetric data loading
# ---------------------------------------------------------------------------
def load_volumetric_data(p):
    """
    Load volumetric (NIfTI) fMRI data and extract metadata.

    Parameters
    ----------
    p : dict
        Path dictionary from set_paths().

    Returns
    -------
    scan_data : ndarray
        4D fMRI data array (x, y, z, time).
    func_img : nib.Nifti1Image
        NIfTI image object (for affine/header).
    metadata : dict
        Extracted metadata: tr_length, voxel_size, nTRs.
    """
    func_img = nib.load(p['pRF_func'])
    scan_data = func_img.get_fdata()
    header = func_img.header

    metadata = {
        'tr_length': _tr_to_sec(float(header['pixdim'][4])),
        'nTRs': scan_data.shape[-1],
    }

    p.update(metadata)  # Update params dictionary with nifti metadata

    return scan_data, func_img


def extract_brainmask_voxels(scan_data, brainmask=None):
    """
    Extract timeseries with variance and their 3D indices from volumetric data. Optionally also apply a brainmask.

    Parameters
    ----------
    scan_data : ndarray
        4D fMRI data (x, y, z, time).
    brainmask : ndarray or None
        Optional 3D binary brainmask. If None, uses non-zero voxels from scan_data.

    Returns
    -------
    timeseries_data : ndarray
        2D array (n_voxels, n_timepoints).
    indices : list of tuple
        List of (x, y, z) index tuples for each voxel.
    """

    # Use voxels with non-zero standard deviation across time
    signal_mask = np.std(scan_data, axis=-1) > 0
    # If a brainmask is provided, further restrict to voxels within the mask
    if brainmask is not None:
        signal_mask &= (brainmask > 0)

    # Extract the indices of the voxels that meet the criteria
    xi, yi, zi = np.nonzero(signal_mask)

    indices = [(xi[i], yi[i], zi[i]) for i in range(len(xi))]
    timeseries_data = scan_data[xi, yi, zi, :]

    return timeseries_data, indices


# ---------------------------------------------------------------------------
# Surface data loading
# ---------------------------------------------------------------------------
def _gifti_meta_lookup(img, key, sources=('image', 'darray')):
    """
    Look up a metadata key from a GIFTI image, checking the requested
    metadata sources in order ('image' -> img.meta, 'darray' -> the first
    data array's meta). Returns None if the key isn't found in any of them.
    """
    lookup = {
        'image': img.meta,
        'darray': img.darrays[0].meta if img.darrays else None,
    }
    for source in sources:
        meta = lookup[source]
        if meta is not None and key in meta:
            return meta[key]
    return None


def load_surface_data(p):
    """
    Load surface (GIFTI) fMRI data and organize it by hemisphere.

    Reads the GIFTI file(s) at p['pRF_func'] (set by set_paths(); one or two
    files) and assigns each to 'left'/'right' based on its
    AnatomicalStructurePrimary metadata. Also extracts nTRs and TR length
    (from 'TimeStep' metadata) and records them in p['nTRs'] / p['tr_length'].

    Parameters
    ----------
    p : dict
        Path dictionary from set_paths(). Updated in place with 'nTRs' and,
        if found, 'tr_length'.

    Returns
    -------
    hemi_data : dict
        {'left': ndarray, 'right': ndarray}, each (n_vertices, n_timepoints).
        Only the hemispheres actually found are included.
    """
    funcFiles = p['pRF_func']
    if isinstance(funcFiles, str):
        funcFiles = [funcFiles]

    hemi_data = {}
    nTRs_by_hemi = {}
    tr_by_hemi = {}

    hemi_by_struct = {'CortexLeft': 'left', 'CortexRight': 'right'}

    for fpath in funcFiles:
        img = nib.load(fpath)

        anat_struct = _gifti_meta_lookup(img, 'AnatomicalStructurePrimary')
        hemi = hemi_by_struct[anat_struct]
        if hemi is None:
            raise ValueError(
                f"Could not determine hemisphere for '{fpath}': "
                f"AnatomicalStructurePrimary='{anat_struct}'."
            )
        if hemi in hemi_data:
            raise ValueError(
                f"Found more than one GIFTI file for the '{hemi}' hemisphere "
                f"(most recently '{fpath}')."
            )

        data = np.array([darr.data for darr in img.darrays]).T
        hemi_data[hemi] = data
        nTRs_by_hemi[hemi] = data.shape[1]

        tr_value = _gifti_meta_lookup(img, 'TimeStep', sources=('darray', 'image'))
        try:
            tr_value = _tr_to_sec(float(tr_value))
        except (TypeError, ValueError):
            tr_value = None
        if tr_value == 0:
            tr_value = None
        tr_by_hemi[hemi] = tr_value

    # --- nTRs: must agree across hemispheres ---
    unique_nTRs = set(nTRs_by_hemi.values())
    if len(unique_nTRs) > 1:
        raise ValueError(f"nTRs differ across hemispheres: {nTRs_by_hemi}")
    p['nTRs'] = unique_nTRs.pop()

    # --- TR length: found via 'TimeStep' metadata, must agree where present ---
    if all(tr is None for tr in tr_by_hemi.values()):
        warnings.warn(
            "Could not find TR ('TimeStep') metadata in any surface GIFTI file; "
            "p['tr_length'] left unchanged."
        )
    else:
        unique_trs = set(tr_by_hemi.values())
        if len(unique_trs) > 1:
            raise ValueError(f"tr_length differs across hemispheres: {tr_by_hemi}")
        p['tr_length'] = unique_trs.pop()

    return hemi_data


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
    img = nib.nifti1.Nifti1Image(data.astype(np.float32), affine=affine, header=header)
    nib.save(img, fpath)
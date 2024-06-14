import numpy as np
import ctypes
import matplotlib.pyplot as plt
import popeye.og as og
import popeye.utilities as utils
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus
from popeye import css
from scipy.io import loadmat
import time
import nibabel as nib
from nilearn import plotting
import os
import tqdm
import pickle
import multiprocessing as mp
import time

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
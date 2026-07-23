"""
H04_grid_predict.py — Grid prediction generation for the CSS pRF model.

Generates predicted BOLD timeseries for each point in the parameter grid.
This is the most computationally expensive step and results are cached to disk.

Key functions:
    generate_grid_prediction()  — Predict timeseries for one grid point
    getGridPreds()              — Parallel prediction for all grid points
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.signal import fftconvolve

from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries_nomask
import popeye.utilities_cclab as utils


def generate_grid_prediction(args):
    """
    Generate a predicted BOLD timeseries for a single CSS pRF model.

    The model:
        1. Create a 2D Gaussian receptive field at (x, y) with size sigma
        2. Convolve RF with stimulus to get neural response timeseries
        3. Apply CSS compressive nonlinearity (response ** n)
        4. Convolve with double-gamma HRF
        5. Normalize to percent signal change

    Parameters
    ----------
    args : tuple
        (x, y, sigma, n, stimulus) where stimulus is a VisualStimulus object.

    Returns
    -------
    predsig : ndarray
        Predicted BOLD timeseries (n_timepoints,), or None if error.
    """
    try:
        x, y, sigma, n, stimulus = args

        # Generate 2D Gaussian receptive field
        rf = generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
        rf /= ((2 * np.pi * sigma**2) * 1 / np.diff(stimulus.deg_x[0, 0:2])**2)

        # RF × stimulus → neural response timeseries
        response = generate_rf_timeseries_nomask(stimulus.stim_arr, rf)

        # CSS compressive nonlinearity
        response **= n

        # Convolve with HRF
        predsig = fftconvolve(response, utils.double_gamma_hrf(0, 1.3))[0:len(response)]

        # Normalize to percent signal change
        predsig = (predsig - np.mean(predsig)) / np.mean(predsig)

        return predsig

    except Exception as e:
        print(f"Error in generate_grid_prediction: {e}")
        return None


def getGridPreds(grid_space, stimulus, gridPath, nTRs):
    """
    Generate predicted timeseries for all grid points in parallel, and cache to disk.

    Parameters
    ----------
    grid_space : list of tuple
        List of (x, y, sigma, n) grid points.
    stimulus : VisualStimulus
        Popeye stimulus object.
    gridPath : str
        Path to save/load cached grid predictions (.npy).
    nTRs : int
        Number of timepoints (for pre-allocation).

    Returns
    -------
    grid_preds : ndarray
        Array of shape (n_grid_points, nTRs) with predicted timeseries.
    """
    grid_preds = np.empty((len(grid_space), nTRs))
    print(f"Starting prediction generation for {len(grid_space)} grid points...")

    with Pool(cpu_count()) as pool:
        results = pool.map(
            generate_grid_prediction,
            [(x, y, s, n, stimulus) for x, y, s, n in grid_space]
        )

    for i, prediction in enumerate(results):
        grid_preds[i] = prediction

    # Cache to disk
    np.save(gridPath, grid_preds)
    print(f"Grid predictions saved to {gridPath}")

    return grid_preds

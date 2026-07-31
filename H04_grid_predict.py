"""
H04_grid_predict.py — Grid prediction generation for the CSS pRF model.

Generates predicted BOLD timeseries for each point in the parameter grid.
This is the most computationally expensive step and results are cached to disk.

Key functions:
    generate_grid_prediction()  — Predict timeseries for one grid point
    getGridPreds()              — Parallel prediction for all grid points
"""

import numpy as np
import sweepea.utilities as utils
from multiprocessing import Pool, cpu_count


# ---------------------------------------------------------------------------
# CPU path — Pool worker globals (set once via initializer, never pickled)
# ---------------------------------------------------------------------------

_stimulus = None
_hrf      = None

def _worker_init_gridpredict(stimulus, hrf):
    """Set stimulus + bounds once per worker subprocess at Pool startup. Also set HRF."""
    global _stimulus, _hrf
    _stimulus = stimulus
    _hrf = hrf


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
        x, y, sigma, n, stimulus, hrf = args

        # Generate 2D Gaussian receptive field
        rf = utils.generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)

        #We do normalization steps in visual stim and generate_og_receptive_field, so we don't need to do it here
        #rf /= ((2 * np.pi * sigma**2) * 1 / np.diff(stimulus.deg_x[0, 0:2])**2)

        # RF × stimulus → neural response timeseries YOU ARE HERE!!!
        response = utils.generate_rf_timeseries(stimulus.stim_arr, rf)

        # CSS compressive nonlinearity
        response **= n

        # Convolve with HRF
        predsig = np.convolve(response, hrf)[0:len(response)]

        # Normalize to percent signal change [not needed b/c we do regression in fitting]
        # mean_predsig = np.mean(predsig)
        # predsig = (predsig - mean_predsig) / mean_predsig

        return predsig

    except Exception as e:
        print(f"Error in generate_grid_prediction: {e}")
        return None

def generate_grid_pred_pool(args):

    return generate_grid_prediction((*args,_stimulus, _hrf))

def getGridPreds(grid_space, stimulus, gridPath, nTRs, hrf):
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

    n_workers = cpu_count()
    chunksize = max(1, grid_preds.shape[0] // (n_workers * 4))

    with Pool(
            cpu_count(),
            initializer=_worker_init_gridpredict,
            initargs=(stimulus, hrf)
        ) as pool:
        results = pool.map(
            generate_grid_pred_pool,
            [(x, y, s, n) for x, y, s, n in grid_space],chunksize=chunksize
        )

    for i, prediction in enumerate(results):
        grid_preds[i] = prediction

    # Cache to disk
    np.save(gridPath, grid_preds)
    print(f"Grid predictions saved to {gridPath}")

    return grid_preds

"""
H04_grid_predict.py — Grid prediction generation for the CSS pRF model.

Generates predicted BOLD timeseries for each point in the parameter grid.
This is the most computationally expensive step and results are cached to disk.

Key functions:
    generate_grid_prediction()  — Predict timeseries for one grid point
    getGridPreds()              — Parallel prediction for all grid points
"""

import numpy as np
import jax
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
import jax.numpy as jnp
# jax.config.update("jax_debug_nans", True)
# jax.config.update('jax_debug_infs', True)
#jax.config.update("jax_disable_jit", True)
#jax.config.update('jax_platform_name', 'cpu')
#from jax.experimental import checkify
import sweepea.utilities as utils
from scipy.stats import zscore



#CPU version of grid_pred for testing purposes
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

        # zscore so we can simplify all the regression 
        predsig = zscore(predsig)

        # this is a way to deal w/ bad parameter combinations. A better way would be to avoid them 
        # entirely through sensible bounds/constraints
        if not np.isfinite(predsig).all():
            # if np.isfinite(predsig).any():
            #     print('foo!')
            predsig = np.zeros_like(predsig)


        return predsig

    except Exception as e:
        print(f"Error in generate_grid_prediction: {e}")
        return None


def _zscore_pred(x,x_mu,x_std):
    return (x- x_mu) / x_std

def _degenerate_pred(x,*args):
    return jnp.zeros_like(x)

def generate_grid_prediction_jax(params, deg_x, deg_y, stim_arr, hrf):
    """JAX version of prf prediction
    """
    x, y, sigma, n = params

    rf = jnp.exp(-((deg_x - x) ** 2 + (deg_y - y) ** 2) /
                (2.0 * sigma ** 2))

    response = jnp.dot(stim_arr, rf.reshape(-1))

    # jax.lax.cond(jnp.all(response==0),
    #              lambda: jax.debug.print('bad!'), lambda: None)
    #jax.debug.print(ordered=True)("response {x}",x=jnp.any(jnp.any(response>0)))
    # jax.debug.print(ordered=True)("rf {x}",x=jnp.any(jnp.isnan(rf)))
    # jax.debug.print(ordered=True)("resp {x}",x=jnp.any(jnp.isnan(response)))

    #temporary masking of zeros is needed to prevent NaNs in derivative of response**n
    #would posssibly be better to just branch at this point but fine for now
    mask = response == 0
    response = jnp.where(mask, 1, response)
    response = response ** n
    response = jnp.where(mask, 0, response)

    predsig = jnp.convolve(response, hrf, mode="full")[:response.shape[0]]

    pred_mean = jnp.mean(predsig)
    pred_std = jnp.std(predsig, mean=pred_mean)

    return jax.lax.cond(pred_std > 0,_zscore_pred,_degenerate_pred,predsig,pred_mean,pred_std)



def getGridPreds(grid_space, stimulus, hrf, gridPath=None, batch_size=2000):
    """
    Generate predicted timeseries for all grid points in parallel, and cache to disk (optional).

    Parameters
    ----------
    grid_space : list of tuple
        List of (x, y, sigma, n) grid points.
    stimulus : VisualStimulus
        Popeye stimulus object.
    hrf : array-like
        Hemodynamic response function.
    gridPath : str, optional
        Path to save/load cached grid predictions (.npy).
    batch_size : int, optional
        Number of grid points to process in each batch call to lax.map (default: 2000).

    Returns
    -------
    grid_preds : ndarray
        Array of shape (n_grid_points, nTRs) with predicted timeseries.
    """

    #grid_preds = np.empty((len(grid_space), stimulus.run_length), dtype=np.float32)
    print(f"Starting prediction generation for {len(grid_space)} grid points...")

    @jax.jit(static_argnames='batch_size')
    def gen_preds(space,deg_x, deg_y, stim_arr, hrf,batch_size):

        def process_one(params):
            return generate_grid_prediction_jax(params, deg_x, deg_y, stim_arr, hrf)
        
        return jax.lax.map(process_one,space,batch_size=batch_size)

    grid_preds = gen_preds(jnp.asarray(grid_space),jnp.asarray(stimulus.deg_x), 
                               jnp.asarray(stimulus.deg_y), jnp.asarray(stimulus.stim_arr), jnp.asarray(hrf),batch_size)
    grid_preds = jax.device_get(grid_preds)
        
    # Cache to disk
    if gridPath is not None:
        np.save(gridPath, grid_preds)
        print(f"Grid predictions saved to {gridPath}")

    return grid_preds

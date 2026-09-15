"""This module contains various utility methods that support functionality in
other modules.  

NOTE: This should probably eventually be combined w/ H03_fit_utils.py????

"""

import numpy as np
from scipy.stats import gamma
from scipy.signal import detrend
from scipy.integrate import trapezoid
from numba import jit


### DATA PROCESSING HELPER FUNCTIONS ###

def percent_change(ts, ax=-1):

    r"""Returns the % signal change of each point of the times series
    along a given axis of the array timeseries

    Parameters
    ----------
    ts : ndarray
        an array of time series

    ax : int, optional (default to -1)
        the axis of time_series along which to compute means and stdevs

    Returns
    -------
    ndarray
        the renormalized time series array (in units of %)

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> ts = np.arange(4*5).reshape(4,5)
    >>> ax = 0
    >>> percent_change(ts,ax)
    array([[-100.    ,  -88.2353,  -78.9474,  -71.4286,  -65.2174],
           [ -33.3333,  -29.4118,  -26.3158,  -23.8095,  -21.7391],
           [  33.3333,   29.4118,   26.3158,   23.8095,   21.7391],
           [ 100.    ,   88.2353,   78.9474,   71.4286,   65.2174]])
    >>> ax = 1
    >>> percent_change(ts,ax)
    array([[-100.    ,  -50.    ,    0.    ,   50.    ,  100.    ],
           [ -28.5714,  -14.2857,    0.    ,   14.2857,   28.5714],
           [ -16.6667,   -8.3333,    0.    ,    8.3333,   16.6667],
           [ -11.7647,   -5.8824,    0.    ,    5.8824,   11.7647]])
    """
    ts = np.asarray(ts)

    return (ts / np.expand_dims(np.mean(ts, ax), ax) - 1) * 100


def detrend_psc(ts,ax=-1):
    ts_mean = np.mean(ts, axis=ax)[..., None]
    ts_detrend = detrend(ts, axis=ax, type='linear') + ts_mean
    ts_pct = percent_change(ts_detrend, ax=-1)
    return ts_pct


### HRF models ###

def double_gamma_hrf(delay, tr, fptr=1.0, integrator=trapezoid,dtype='float32'):

    r"""The double gamma hemodynamic reponse function (HRF).
    The user specifies only the delay of the peak and undershoot.
    The delay shifts the peak and undershoot by a variable number of
    seconds. The other parameters are hardcoded. The HRF delay is
    modeled for each voxel independently. The form of the HRF and the
    hardcoded values are based on previous work [1]_.

    Parameters
    ----------
    delay : float
        The delay of the HRF peak and undershoot.

    tr : float
        The length of the repetition time in seconds.

    fptr : float
        The number of stimulus frames per reptition time.  For a
        60 Hz projector and with a 1 s repetition time, the fptr
        would be equal to 60.  It is possible that you will bin all
        the frames in a single TR, in which case fptr equals 1.

    integrator : callable
        The integration function for normalizing the units of the HRF
        so that the area under the curve is the same for differently
        delayed HRFs.  Set integrator to None to turn off normalization.

    Returns
    -------
    hrf : ndarray
        The hemodynamic response function to convolve with the stimulus
        timeseries.

    Reference
    ----------
    .. [1] Glover, GH (1999) Deconvolution of impulse response in event related
    BOLD fMRI. NeuroImage 9, 416-429.

    """
    from scipy.special import gamma
    
    # add delay to the peak and undershoot params (alpha 1 and 2)
    alpha_1 = float(5 + delay)
    beta_1 = 1.0
    c = 0.1
    alpha_2 = float(15 + delay)
    beta_2 = 1.0
    
    t = np.arange(0,32,tr)
    
    hrf = ( ( ( t ** (alpha_1) * beta_1 ** alpha_1 * np.exp( -beta_1 * t )) /gamma( alpha_1 )) - c *
            ( ( t ** (alpha_2) * beta_2 ** alpha_2 * np.exp( -beta_2 * t )) /gamma( alpha_2 )) )
            
    if integrator: # pragma: no cover
        hrf /= integrator(hrf)
        
    return hrf.astype(dtype)

def _gamma_difference_hrf(tr, oversampling=1, time_length=32., onset=0.,
                         delay=5, undershoot=15., dispersion=1.,
                         u_dispersion=1., ratio=0.167):
    """ Compute an hrf as the difference of two gamma functions
    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the hrf
    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length, int(float(time_length) / dt))
    time_stamps -= onset / dt
    hrf = gamma.pdf(time_stamps, delay / dispersion, dt / dispersion) - \
        ratio * gamma.pdf(
        time_stamps, undershoot / u_dispersion, dt / u_dispersion)
    hrf /= trapezoid(hrf)
    return hrf

def spm_hrf(delay, tr, oversampling=1, time_length=32., onset=0.):
    """ Implementation of the SPM hrf model
    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response
    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    return _gamma_difference_hrf(tr, oversampling, time_length, onset, delay=5+delay, undershoot=15+delay,)


def glover_hrf(delay, tr, oversampling=1, time_length=32., onset=0.):
    """ Implementation of the Glover hrf model
    Parameters
    ----------
    tr: float, scan repeat time, in seconds
    oversampling: int, temporal oversampling factor, optional
    time_length: float, hrf kernel length, in seconds
    onset: float, onset of the response
    Returns
    -------
    hrf: array of shape(length / tr * oversampling, float),
         hrf sampling on the oversampled time grid
    """
    return _gamma_difference_hrf(tr, oversampling, time_length, onset,
                                delay=5+delay, undershoot=15+delay, dispersion=.9,
                                u_dispersion=.9, ratio=.35)


### PRF MODEL HELPER FUNCTIONS ###

@jit(nopython=True,parallel=False)
def generate_og_receptive_field_2d(x,y,sigma,deg_x,deg_y):
    
    d = (deg_x-x)**2 + (deg_y-y)**2
        
    rf = np.exp(-d / (2.0 * sigma**2))
    
    return rf

def generate_og_receptive_field(x,y,sigma,deg_x,deg_y):
    #return rf in 1D for matrix multiplication. Could     
        
    rf = generate_og_receptive_field_2d(x,y,sigma,deg_x,deg_y)
    
    return np.reshape(rf,(rf.shape[0]*rf.shape[1],1)).astype(np.float32)


#have not found that this benefits from numba on local machine...
def generate_rf_timeseries(stim_arr,rf):
    return np.squeeze(np.dot(stim_arr,rf))


### HELPER FUNCTIONS FOR FITTING ###
#NOTE: Currently deprecated, but possible some are worth revisiting in case Numba speedup would overcome function call.

def make_dmat(ts):
    return np.column_stack((ts,np.ones(ts.shape,dtype=np.float32)))


@jit(nopython=True, parallel=False)
def rss(data,prediction):
    d = data - prediction
    return d.dot(d)

def error_function_rss(parameters, data, objective_function,verbose):
    prediction = objective_function(*parameters)
    d = data - prediction
    error = d.dot(d)
    #error = np.sum((data-prediction)**2)
    
    #return something very large if we encounter bad values
    if np.isfinite(error):
        return error
    else:
        d = data - data.mean()
        return d.dot(d)*1e10

def error_function_lsq(parameters, data, objective_function,verbose):
    
    prediction = objective_function(*parameters)
    dmat = np.column_stack((prediction,np.ones(prediction.shape)))
    sol = np.linalg.lstsq(dmat,data,rcond=None)
    error = sol[1][0]
    #return something very large if we encounter bad values
    
    if np.isfinite(error):
        if sol[0][0] >= 0: #check beta for constraint
            return error
        else:
            #given our regression equation, the minimial constrained least squares solution is 
            #beta = 0, intercept = mean...e.g. the error is the rss of the data
            d = data - data.mean()
            return d.dot(d) - sol[0][0] # let's help out gradient w/ L1 penality on negative beta
    else:
        d = data - data.mean()
        return d.dot(d)*1e10
    
    
@jit(nopython=True,parallel=False)
def do_lsq_error(prediction, data, rawrss):
    
    #minimize algorithms don't obey constraints no matter how hard I try...
    if np.allclose(prediction,0):
        return rawrss*1e10
    
    dmat = np.column_stack((prediction,np.ones(prediction.shape,dtype=np.float32)))
    try:
        sol = np.linalg.lstsq(dmat,data)
    except Exception:
        #return something very large if we encounter bad values
        return rawrss*1e10
    error = sol[1][0]
    
    if sol[0][0] >= 0: #check beta for constraint
        return error
    else:
        #given our regression equation, the minimial constrained least squares solution is 
        #beta = 0, intercept = mean...e.g. the error is the rss of the data
        return rawrss - sol[0][0]
        #d = data - data.mean()
        #return d.dot(d) - sol[0][0] # let's help out gradient w/ L1 penality on negative beta
    
    # if np.isfinite(error):
    #     if sol[0][0] >= 0: #check beta for constraint
    #         return error
    #     else:
    #         #given our regression equation, the minimial constrained least squares solution is 
    #         #beta = 0, intercept = mean...e.g. the error is the rss of the data
    #         return rawrss - sol[0][0]
    #         #d = data - data.mean()
    #         #return d.dot(d) - sol[0][0] # let's help out gradient w/ L1 penality on negative beta
    # else:
    #     return rawrss*1e10
    

### MODEL constratin functions ###
#NOTE: currently not used and implemented elsewhere only for grid construction, but leaving in just in for now

@jit(nopython=True,parallel=False)    
def dist_con(x):
    return np.sqrt(x[0]**2 + x[1]**2)
    
@jit(nopython=True,parallel=False)   
def prfsize_con(x,outer_limit):
    return np.sqrt(x[0]**2 + x[1]**2) - outer_limit*x[2]
 







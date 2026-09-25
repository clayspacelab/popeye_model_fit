"""
Class and functions for processing prf stimuli. Adapted from popeye visual_stimulus.py.

First pass at a stimulus model for abstracting the qualities and functionality of a stimulus
into an abstract class.  For now, we'll assume the stimulus model only pertains to visual 
stimuli on a visual display over time (i.e., 3D).  Hopefully this can be extended to other stimuli
with an arbitrary number of dimensions (e.g., auditory stimuli).

"""

import numpy as np
from collections import namedtuple
from scipy.ndimage.interpolation import zoom


def dva(stim_width, viewing_distance):
    """Computes the visual angle of the stimulus in degrees."""
    #https://www.sr-research.com/eye-tracking-blog/background/visual-angle/
    return 2 * np.degrees(np.arctan(stim_width / (2 * viewing_distance)))

# def degrees_per_gridpoint(gridpoints_across, stim_width, viewing_distance):
#     """Computes the number of degrees of visual angle per grid point."""    
#     return dva(stim_width, viewing_distance) / gridpoints_across

def degrees_per_gridpoint(stim_dva, gridpoints_across):
    """Computes the number of degrees of visual angle per grid point."""    
    return stim_dva / gridpoints_across


def stim2d(stim_arr):
    stim_arr_long = stim_arr.transpose(2,0,1)
    stim_arr_long = np.reshape(stim_arr_long,(stim_arr_long.shape[0],-1))
    
    return stim_arr_long
    
def generate_coordinate_matrices(gridpoints_across, gridpoints_down, dpp, scale_factor=1, dtype=np.float32):
    
    """Creates coordinate matrices for representing the visual field in terms
       of degrees of visual angle.
       
    This function takes the screen dimensions, the grid points per degree, and a
    scaling factor in order to generate a pair of ndarrays representing the
    horizontal and vertical extents of the stimulus in degrees of visual
    angle.
    
    Parameters
    ----------
    gridpoints_across : int
        The number of grid points along the horizontal extent of the visual stimulus.
    gridpoints_down : int
        The number of grid points along the vertical extent of the visual stimulus.
    dpp: float
        The number of degrees of visual angle that spans 1 grid point.  This number
        is computed using the stimulus width and the viewing distance.  See the
        config.init_config for details. 
    scale_factor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.
    dtype : numpy dtype, optional
        Datatype for the returned arrays.
        
    Returns
    -------
    deg_x : ndarray
        An array representing the horizontal extent of the visual stimulus in
        terms of degrees of visual angle.
    deg_y : ndarray
        An array representing the vertical extent of the visual stimulus in
        terms of degrees of visual angle.
    """
    
    [X,Y] = np.meshgrid(np.arange(np.round(gridpoints_across*scale_factor)),
                        np.arange(np.round(gridpoints_down*scale_factor)))
                        
                        
    deg_x = (X-np.round(gridpoints_across*scale_factor)/2)*(dpp*scale_factor)
    deg_y = (Y-np.round(gridpoints_down*scale_factor)/2)*(dpp*scale_factor)
    
    deg_x += 0.5*(dpp*scale_factor)
    deg_y += 0.5*(dpp*scale_factor)
    
    return deg_x.astype(dtype), np.flipud(deg_y).astype(dtype)


def resample_stimulus(stim_arr, scale_factor=0.05, mode='nearest',
                      order=0, dtype=np.float32):
    
    """Resamples the visual stimulus
    
    The function takes an ndarray `stim_arr` and resamples it by the user
    specified `scale_factor`.  The stimulus array is assumed to be a three
    dimensional ndarray representing the stimulus, in screen pixel coordinates,
    over time.  The first two dimensions of `stim_arr` together represent the
    exent of the visual stimulus (grid points) and the last dimensions represents
    time (TRs).

    The underlying function used here is `scipy.ndimage.zoom`. Some arguments
    are passed through to that function.
    
    Parameters
    ----------
    stim_arr : ndarray
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.
    
    scale_factor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.
    
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'nearest'.

    order : int, optional
        Interpolation order, must be in range 0-5.

    dtype : numpy dtype, optional
        Datatype for the returned array.
        
    Returns
    -------
    resampled_arr : ndarray
        An array that is resampled according to the user-specified scale factor.
    """
    
    dims = np.shape(stim_arr)
    resampled_arr = np.zeros((int(round(dims[0] * scale_factor)),
                              int(round(dims[1] * scale_factor)),
                              dims[2]), dtype=dtype)
    
    # loop
    for tr in np.arange(dims[-1]):
        
        # resize it
        f = zoom(stim_arr[:,:,tr], scale_factor, mode=mode, order=order)
        
        # insert it
        resampled_arr[:,:,tr] = f
    
    return resampled_arr


#define a named tuple for packaging up the stimulus parameters we're actually using for fitting in 
#VisualStimulus.
_StimParams = namedtuple('_StimParams', ['stim_arr', 'deg_x', 'deg_y', 'run_length'])

class VisualStimulus:
    
    
    def __init__(self, stim_arr, tr_length, viewing_distance=None, stim_width=None, stim_dva=None,
                 scale_factor=1.0, nTRs=None, dtype=np.float32, interp='nearest'):
        
        """
        Parameters
        ----------
        stim_arr : ndarray
            An array containing the visual stimulus (x,y,time).
        tr_length : float
            repitition time (seconds).
        viewing_distance : float, optional
            The distance between the participant and the display. Must be
            given together with `stim_width.
        stim_width : float, optional
            The width of the stimulus. This is used together with
            `viewing_distance` to compute the visual angle. Must be given together with
            `viewing_distance`; leave both unset and pass `stim_dva` directly
            instead if the visual angle is already known.
        stim_dva : float, optional
            The width of the stimulus in degrees of visual angle. Provide this
            directly as an alternative to `viewing_distance`/`stim_width`. 
        scale_factor : float, optional
            The downsampling rate for ball-parking a solution. (DEPRECATED)
            Default is 1.0 (no downsampling).
        nTRs : int, optional
            The number of TRs in the functional data, used as a sanity check
            that the stimulus and functional run lengths match. 
        dtype : numpy dtype, optional
            Datatype for the stored stimulus and coordinate arrays. Wouldn't advise changing.
        interp : str, optional
            Interpolation mode passed to `resample_stimulus` when downsampling
            (see `scipy.ndimage.zoom`'s `mode` argument). Default is 'nearest'. (DEPRECATED)
        """

        if stim_dva is not None:
            if viewing_distance is not None or stim_width is not None:
                raise ValueError(
                    "Provide either stim_dva, or viewing_distance and stim_width, not both.")
        else:
            if viewing_distance is None or stim_width is None:
                raise ValueError(
                    "Provide stim_dva, or both viewing_distance and stim_width.")
            stim_dva = dva(stim_width, viewing_distance)

        
        # absorb the vars
        self.dtype = dtype
        self.stim_arr = np.array(stim_arr,dtype=self.dtype) 
        self.tr_length = tr_length
        self.viewing_distance = viewing_distance  # may be None
        self.stim_width = stim_width          # may be None
        self.stim_dva = stim_dva
        self.scale_factor = scale_factor
        self.interp = interp
        
        # ascertain stimulus features
        self.gridpoints_across = self.stim_arr.shape[1]
        self.gridpoints_down = self.stim_arr.shape[0]
        self.dpp = degrees_per_gridpoint(self.stim_dva,self.gridpoints_across)

        #At this point we don't allow stim files w/ timing that deviates from the BOLD data, but nTRs from the BOLD
        #can be passed as a sanity check that BOLD and stimulus run lengths match
        if (nTRs is not None) and (nTRs != self.stim_arr.shape[2]):
            raise ValueError(f"Stimulus TRs ({self.stim_arr.shape[2]}) does not match functional data TRs ({nTRs}).")
        else:
            self.run_length = self.stim_arr.shape[2]
        
        #we also want screen width in dva for computing constraints on fits
        #self.stim_dva = dva(self.stim_width, self.viewing_distance)

        # generate coordinate matrices
        self.deg_x, self.deg_y = generate_coordinate_matrices(self.gridpoints_across, 
                                                              self.gridpoints_down, self.dpp, dtype=self.dtype)

        #convert stim_arr to 2d for faster processing
        self.stim_arr = stim2d(self.stim_arr)
        
        if self.scale_factor == 1.0:
            
            self.stim_arr0 = self.stim_arr
            self.deg_x0 = self.deg_x
            self.deg_y0 = self.deg_y
            
        else:
            
            # create downsampled stimulus
            stim_arr0 = resample_stimulus(self.stim_arr, self.scale_factor, mode=self.interp, dtype=self.dtype)
            self.stim_arr0 = stim2d(stim_arr0)
            
            # generate the coordinate matrices
            self.deg_x0, self.deg_y0 = generate_coordinate_matrices(self.gridpoints_across, self.gridpoints_down, self.dpp, 
                                                          self.scale_factor, dtype=self.dtype)
            
        
        # add dpp for the down-sampled stimulus
        self.dpp0 = degrees_per_gridpoint(self.stim_dva, self.gridpoints_across * self.scale_factor)
        
        # rescale stim grids according to dpp 
        # (this roughly follows Vista approach to give iterpretable betas in terms of psc as a function of 
        # size of stimulus, but mostly doing it for numerical reasons to keep response range consistent/in check)
        self.stim_arr *= self.dpp**2
        self.stim_arr0 *= self.dpp0**2

        #finally package up parameters we're actually using for slimmer memory footprint during fitting
        #really this whole scheme should be rethought, but for now this is a quick fix to avoid passing around a 
        #huge stimulus class object
        self.params = _StimParams(stim_arr=self.stim_arr, deg_x=self.deg_x, 
                                 deg_y=self.deg_y, run_length=self.run_length)
        
        

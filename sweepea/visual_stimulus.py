"""

First pass at a stimulus model for abstracting the qualities and functionality of a stimulus
into an abstract class.  For now, we'll assume the stimulus model only pertains to visual 
stimuli on a visual display over time (i.e., 3D).  Hopefully this can be extended to other stimuli
with an arbitrary number of dimensions (e.g., auditory stimuli).

"""

import numpy as np
from collections import namedtuple
from scipy.ndimage.interpolation import zoom



def pixels_per_degree(pixels_across, screen_width, viewing_distance):
    """Computes the number of pixels per degree of visual angle."""    
    return np.pi*pixels_across/np.arctan(screen_width/viewing_distance/2.0)/360.0


def stim2d(stim_arr):
    stim_arr_long = stim_arr.transpose(2,0,1)
    stim_arr_long = np.reshape(stim_arr_long,(stim_arr_long.shape[0],-1))
    
    return stim_arr_long
    
def generate_coordinate_matrices(pixels_across, pixels_down, ppd, scale_factor=1, dtype=np.float32):
    
    """Creates coordinate matrices for representing the visual field in terms
       of degrees of visual angle.
       
    This function takes the screen dimensions, the pixels per degree, and a
    scaling factor in order to generate a pair of ndarrays representing the
    horizontal and vertical extents of the visual display in degrees of visual
    angle.
    
    Parameters
    ----------
    pixels_across : int
        The number of pixels along the horizontal extent of the visual display.
    pixels_down : int
        The number of pixels along the vertical extent of the visual display.
    ppd: float
        The number of pixels that spans 1 degree of visual angle.  This number
        is computed using the display width and the viewing distance.  See the
        config.init_config for details. 
    scale_factor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.
    dtype : numpy dtype, optional
        Datatype for the returned arrays.
        
    Returns
    -------
    deg_x : ndarray
        An array representing the horizontal extent of the visual display in
        terms of degrees of visual angle.
    deg_y : ndarray
        An array representing the vertical extent of the visual display in
        terms of degrees of visual angle.
    """
    
    [X,Y] = np.meshgrid(np.arange(np.round(pixels_across*scale_factor)),
                        np.arange(np.round(pixels_down*scale_factor)))
                        
                        
    deg_x = (X-np.round(pixels_across*scale_factor)/2)/(ppd*scale_factor)
    deg_y = (Y-np.round(pixels_down*scale_factor)/2)/(ppd*scale_factor)
    
    deg_x += 0.5/(ppd*scale_factor)
    deg_y += 0.5/(ppd*scale_factor)
    
    return deg_x.astype(dtype), np.flipud(deg_y).astype(dtype)


def resample_stimulus(stim_arr, scale_factor=0.05, mode='nearest',
                      order=0, dtype=np.float32):
    
    """Resamples the visual stimulus
    
    The function takes an ndarray `stim_arr` and resamples it by the user
    specified `scale_factor`.  The stimulus array is assumed to be a three
    dimensional ndarray representing the stimulus, in screen pixel coordinates,
    over time.  The first two dimensions of `stim_arr` together represent the
    exent of the visual display (pixels) and the last dimensions represents
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
    
    
    def __init__(self, stim_arr, viewing_distance, screen_width,
                 scale_factor, tr_length, dtype=np.float32, interp='nearest'):
        
        """
         
        
        Paramaters
        ----------
        
        stim_arr : ndarray
            An array containing the visual stimulus at the native resolution. The 
            visual stimulus is assumed to be three-dimensional (x,y,time).
        
        viewing_distance : float
            The distance between the participant and the display (cm).
            
        screen_width : float
            The width of the display (cm). This is used to compute the visual angle
            for determining the pixels per degree of visual angle.
        
        scale_factor : float
            The downsampling rate for ball=parking a solution. The `stim_arr` is
            downsampled so as to speed up the fitting procedure.  The final model
            estimates will be derived using the non-downsampled stimulus.
            
        """
        
        # absorb the vars
        self.dtype = dtype
        self.stim_arr = np.array(stim_arr,dtype=self.dtype) 
        self.tr_length = tr_length
        self.viewing_distance = viewing_distance
        self.screen_width = screen_width
        self.scale_factor = scale_factor
        self.interp = interp
        
        # ascertain stimulus features
        self.pixels_across = self.stim_arr.shape[1]
        self.pixels_down = self.stim_arr.shape[0]
        self.run_length = self.stim_arr.shape[2]
        self.ppd = pixels_per_degree(self.pixels_across, self.screen_width, self.viewing_distance)
        
        #we also want screen width in dva for computing constraints on fits
        self.screen_dva = self.pixels_across/self.ppd 

        
        # generate coordinate matrices
        self.deg_x, self.deg_y = generate_coordinate_matrices(self.pixels_across, 
                                                              self.pixels_down, self.ppd, dtype=self.dtype)

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
            self.deg_x0, self.deg_y0 = generate_coordinate_matrices(self.pixels_across, self.pixels_down, self.ppd, 
                                                          self.scale_factor, dtype=self.dtype)
            
        
        # add ppd for the down-sampled stimulus
        self.ppd0 = pixels_per_degree(self.pixels_across*self.scale_factor, self.screen_width, self.viewing_distance)
        
        # rescale stim grids according to ppd 
        # (this roughly follows Vista approach to give iterpretable betas in terms of psc as a function of 
        # size of stimulus, but mostly doing it for numerical reasons to keep response range consistent/in check)
        self.stim_arr /= self.ppd**2
        self.stim_arr0 /= self.ppd0**2

        #finally package up parameters we're actually using for slimmer memory footprint during fitting
        #really this whole scheme should be rethought, but for now this is a quick fix to avoid passing around a 
        #huge stimulus class object
        self.params = _StimParams(stim_arr=self.stim_arr, deg_x=self.deg_x, 
                                 deg_y=self.deg_y, run_length=self.run_length)
        
        

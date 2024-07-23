import numpy as np
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries, generate_rf_timeseries_nomask
import popeye.utilities_cclab as utils
from scipy.signal import fftconvolve
from scipy.stats import linregress


def distance_mask_coarse_me(x, y, sigma, stimulus):
        
        distance = (stimulus.deg_x0 - x)**2 + (stimulus.deg_y0 - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < 5*sigma**2] = 1
        
        return mask

def distance_mask_me(x, y, sigma, stimulus):

        distance = (stimulus.deg_x - x)**2 + (stimulus.deg_y - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < 5*sigma**2] = 1

        return mask

def generate_ballpark_prediction_me(x, y, sigma, n, data, stimulus, unscaled=False):
        
        # mask = distance_mask_coarse_me(x, y, sigma, stimulus)

        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma,stimulus.deg_x0, stimulus.deg_y0)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1/np.diff(stimulus.deg_x0[0,0:2])**2)
        
        # extract the stimulus time-series
        # response = generate_rf_timeseries(stimulus.stim_arr0, rf, mask)
        response = generate_rf_timeseries_nomask(stimulus.stim_arr0, rf)
        
        # compression
        response **= n
        
        # convolve with the HRF
        # hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        # # convolve it with the stimulus
        # model = fftconvolve(response, hrf)[0:len(response)]
        model = fftconvolve(response, utils.double_gamma_hrf(0, 1.3))[0:len(response)]
        
        # units
        # model = self.normalizer(model)

        # units
        model = (model - np.mean(model)) / np.mean(model)
        
        if unscaled:
            return model
        else:
            # regress out mean and linear
            p = linregress(model, data)
            # print(p)
            
            # scale
            model *= p[0]
            
            # offset
            model += p[1]
            
            return model

def generate_prediction_me(x, y, sigma, n, beta, baseline, stimulus, unscaled=False):
        
        # mask = distance_mask_me(x, y, sigma, stimulus)

        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma, stimulus.deg_x, stimulus.deg_y)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1/np.diff(stimulus.deg_x[0,0:2])**2)
        
        # extract the stimulus time-series
        # response = generate_rf_timeseries(stimulus.stim_arr, rf, mask)
        response = generate_rf_timeseries_nomask(stimulus.stim_arr, rf)
        
        # compression
        response **= n
        
        # convolve with the HRF
        # hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # # convolve it with the stimulus
        # model = fftconvolve(response, hrf)[0:len(response)]
        model = fftconvolve(response, utils.double_gamma_hrf(0, 1.3))[0:len(response)]
        
        # units
        # model = self.normalizer(model)
        
        # convert units
        model = (model - np.mean(model)) / np.mean(model)
        
        if unscaled:
            return model
        else:
            
            # scale it by beta
            model *= beta
            
            # offset
            model += baseline
            
            return model
        
def generate_prediction_blah(x, y, sigma, n, beta, stimulus, unscaled=False):
        
        return generate_prediction_me(x, y, sigma, n, beta, 0, stimulus, unscaled)
        
def error_function_rss_me(parameters, data, stimulus, objective_function, verbose):
    prediction = objective_function(*parameters, data, stimulus)
    error = utils.rss(data, prediction)
    return error

def error_function_rss_blah(parameters, data, stimulus, objective_function, verbose):
    prediction = objective_function(*parameters, stimulus)
    error = utils.rss(data, prediction)
    return error
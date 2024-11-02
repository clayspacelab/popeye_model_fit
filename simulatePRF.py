import ctypes, random, pickle
import numpy as np
import matplotlib.pyplot as plt

import popeye.utilities_cclab as utils
from popeye.visual_stimulus import VisualStimulus

import nibabel as nib
# from ipywidgets import interact, widgets
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from itertools import product
from scipy.optimize import brute, fmin, minimize, least_squares, fmin_powell
from scipy.io import savemat
from dataloader import *
from fit_utils import *
from fitutils_css import generate_grid_prediction

def main():
    # Initialize parameters
    params = {}
    params['subjID'] = 'JC'
    # Got these from Zhengang, and he got it from rsvp_params.txt
    params['viewingDistance'] = 83.5 #63 #83.5 # in cm
    params['screenWidth'] = 36.2 #35 #36.2 # in cm
    params['scaleFactor'] = 1
    params['resampleFactor'] = 1
    params['dtype'] = ctypes.c_int16

    p = set_paths(params)

    # Load stimulus
    bar, stim_params = load_stimuli(p)
    bar = bar[:, :, 0:201]
    # Mirror y axis (this is done because popeye flips the y axis)
    bar = np.flip(bar, axis=0)

    # Create the visual stimulus
    stimulus = VisualStimulus(stim_arr=bar,
                            viewing_distance=params['viewingDistance'], 
                            screen_width=params['screenWidth'], 
                            scale_factor=params['scaleFactor'], 
                            tr_length=1.3, 
                            dtype=ctypes.c_int16)


    Ns = 100
    x_space = np.concatenate((np.linspace(-stimulus.deg_x0.max(), stimulus.deg_x0.max(), Ns//2),
                        np.geomspace(-stimulus.deg_x0.max(), -2*stimulus.deg_x0.max(), Ns//4),
                            np.geomspace(stimulus.deg_x0.max(), 2*stimulus.deg_x0.max(), Ns//4)))
    y_space = np.concatenate((np.linspace(-stimulus.deg_y0.max(), stimulus.deg_y0.max(), Ns//2),
                            np.geomspace(-stimulus.deg_y0.max(), -2*stimulus.deg_y0.max(), Ns//4),
                            np.geomspace(stimulus.deg_y0.max(), 2*stimulus.deg_y0.max(), Ns//4)))
    s_space = np.concatenate((np.linspace(0.1, 5, 3*Ns//4), np.geomspace(5, stimulus.deg_x0.max(), Ns//4)))
    n_space = np.linspace(0.01, 1, Ns)
    params_space_orig = list(product(x_space, y_space, s_space, n_space))
    params_space = constraint_grids(params_space_orig, stimulus)

    nvoxs = 10000
    params_vox = random.sample(params_space, nvoxs)
    baseline_vox = np.random.uniform(0,10000, nvoxs)

    # Generate predictions for all voxels
    with Pool(cpu_count()) as pool:
        results = pool.map(generate_grid_prediction, [(x, y, s, n, stimulus) for x, y, s, n in params_vox])

    # Store the predictions in an array
    results = np.array(results)
    # Add a random baseline and noise to each voxel and a linear trend
    # Add noise + linear trend + baseline
    results = results + np.random.normal(0, 0.2, results.shape) + np.linspace(0, 1, results.shape[-1])[None, :] + baseline_vox[:, None]#+ np.random.uniform(0, 10000, results.shape[0])[:, None] 

    # Save the results as well as params_space in pickle
    pickle.dump(results, open(os.path.join(p['pRF_data'], 'Simulation', 'simulatedVoxels.pkl'), 'wb'))
    with open(os.path.join(p['pRF_data'], 'Simulation', 'simulatedVoxels.pkl'), 'rb') as f:
        data_temp = pickle.load(f)
    savemat(os.path.join(p['pRF_data'], 'Simulation', 'simulatedVoxels.mat'), {'simulatedVoxels': data_temp})
    # Save both params_vox and baseline_vox
    params_to_save = {'params_vox': params_vox, 'baseline_vox': baseline_vox}
    pickle.dump(params_to_save, open(os.path.join(p['pRF_data'], 'Simulation', 'simulatedParams.pkl'), 'wb'))
    # pickle.dump(params_vox, open(os.path.join(p['pRF_data'], 'Simulation', 'simulatedParams.pkl'), 'wb'))


    # Plot 10 random voxels
    f, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(10):
        ax = axs[i//5, i%5]
        ax.plot(results[i])
        ax.set_title(f'x: {round(params_vox[i][0], 2)}, y: {round(params_vox[i][1], 2)}, s: {round(params_vox[i][2], 2)}, n: {round(params_vox[i][3], 2)}')
    plt.savefig(os.path.join(p['pRF_data'], 'Simulation/figures/simulated_voxels.png'), dpi=300)
    plt.close(f)




if __name__ == '__main__':
    main()

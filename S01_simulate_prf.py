"""
S01_simulate_prf.py — Generate synthetic pRF data with known ground-truth parameters.

Creates simulated voxel timeseries by:
    1. Sampling random (x, y, sigma, n) parameters from the constrained grid space
    2. Generating model predictions for each voxel
    3. Adding noise, linear trend, and baseline
    4. Saving the simulated data and ground-truth parameters

The output can be used with S02_run_simulation_fit.py to validate the pipeline.

Usage:
    python S01_simulate_prf.py
    python S01_simulate_prf.py --n-voxels 5000
"""

import argparse
import ctypes
import random
import pickle
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import savemat
from itertools import product

import popeye.utilities_cclab as utils
from popeye.visual_stimulus import VisualStimulus

from H01_config import DEFAULT_PARAMS, set_paths
from H02_dataloader import load_stimuli
from H03_fit_utils import constraint_grids, set_dark_theme
from H04_grid_predict import generate_grid_prediction


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic pRF data')
    parser.add_argument('--n-voxels', type=int, default=10000,
                        help='Number of simulated voxels (default: 10000)')
    parser.add_argument('--grid-density', type=int, default=100,
                        help='Grid density for parameter sampling (default: 100)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Use a dummy subject to get paths (stimulus is shared)
    params = dict(DEFAULT_PARAMS)
    params['subjID'] = 'JC'
    p, _ = set_paths(params['subjID'], data_format='volumetric')

    # Load stimulus
    bar, stim_params = load_stimuli(p)
    bar = bar[:, :, 0:201]
    bar = np.flip(bar, axis=0)

    # Create stimulus object
    stimulus = VisualStimulus(
        stim_arr=bar,
        viewing_distance=params['viewingDistance'],
        screen_width=params['screenWidth'],
        scale_factor=params['scaleFactor'],
        tr_length=1.3,
        dtype=ctypes.c_int16,
    )

    # Build parameter space
    Ns = args.grid_density
    x_space = np.concatenate((
        np.linspace(-stimulus.deg_x0.max(), stimulus.deg_x0.max(), Ns // 2),
        np.geomspace(-stimulus.deg_x0.max(), -2 * stimulus.deg_x0.max(), Ns // 4),
        np.geomspace(stimulus.deg_x0.max(), 2 * stimulus.deg_x0.max(), Ns // 4),
    ))
    y_space = np.concatenate((
        np.linspace(-stimulus.deg_y0.max(), stimulus.deg_y0.max(), Ns // 2),
        np.geomspace(-stimulus.deg_y0.max(), -2 * stimulus.deg_y0.max(), Ns // 4),
        np.geomspace(stimulus.deg_y0.max(), 2 * stimulus.deg_y0.max(), Ns // 4),
    ))
    s_space = np.concatenate((
        np.linspace(0.1, 5, 3 * Ns // 4),
        np.geomspace(5, stimulus.deg_x0.max(), Ns // 4),
    ))
    n_space = np.linspace(0.01, 1, Ns)

    params_space_orig = list(product(x_space, y_space, s_space, n_space))
    params_space = constraint_grids(params_space_orig, stimulus)

    # Sample random voxels
    nvoxs = args.n_voxels
    params_vox = random.sample(params_space, min(nvoxs, len(params_space)))
    baseline_vox = np.random.uniform(0, 10000, len(params_vox))

    # Generate predictions
    print(f"Generating predictions for {len(params_vox)} simulated voxels...")
    from multiprocessing import Pool, cpu_count
    with Pool(cpu_count()) as pool:
        results = pool.map(
            generate_grid_prediction,
            [(x, y, s, n, stimulus) for x, y, s, n in params_vox]
        )

    results = np.array(results)

    # Add noise + linear trend + baseline
    results = (results
               + np.random.normal(0, 0.2, results.shape)
               + np.linspace(0, 1, results.shape[-1])[None, :]
               + baseline_vox[:, None])

    # Save output
    sim_dir = os.path.join(p['pRF_data'], 'Simulation')
    os.makedirs(sim_dir, exist_ok=True)
    fig_dir = os.path.join(sim_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Save simulated voxels
    pickle.dump(results, open(os.path.join(sim_dir, 'simulatedVoxels.pkl'), 'wb'))
    savemat(os.path.join(sim_dir, 'simulatedVoxels.mat'),
            {'simulatedVoxels': results})

    # Save ground-truth parameters
    params_to_save = {'params_vox': params_vox, 'baseline_vox': baseline_vox}
    pickle.dump(params_to_save,
                open(os.path.join(sim_dir, 'simulatedParams.pkl'), 'wb'))

    print(f"Saved {len(params_vox)} simulated voxels to {sim_dir}")

    # Plot sample voxels with dark theme
    set_dark_theme()
    f, axs = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(10):
        ax = axs[i // 5, i % 5]
        ax.plot(results[i], color='#00e5ff', linewidth=1.5)
        ax.set_title(
            f'x:{params_vox[i][0]:.1f} y:{params_vox[i][1]:.1f} '
            f's:{params_vox[i][2]:.1f} n:{params_vox[i][3]:.2f}',
            color='#ffffff'
        )
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'simulated_voxels.png'), dpi=300)
    plt.close(f)
    print(f"Sample plot saved to {fig_dir}/simulated_voxels.png")


if __name__ == '__main__':
    main()

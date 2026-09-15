"""
S01_simulate_prf.py — Generate synthetic pRF data with known ground-truth parameters.

Creates simulated voxel timeseries by:
    1. Sampling random (x, y, sigma, n) parameters from the constrained grid space
    2. Generating model predictions for each voxel (CPU multiprocessing or GPU batch)
    3. Adding noise, linear trend, and baseline
    4. Saving the simulated data and ground-truth parameters

The output can be used with S02_run_simulation_fit.py to validate the pipeline.

Usage:
    python S01_simulate_prf.py
    python S01_simulate_prf.py --n-voxels 100000
    python S01_simulate_prf.py --n-voxels 100000 --use-gpu
"""

import argparse
import random
import pickle
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.io import savemat
from scipy.stats import zscore
from itertools import product

from config import DEFAULT_PARAMS, GRID_DEFAULTS, set_paths

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic pRF data')
    parser.add_argument('--n-voxels', type=int, default=100000,
                        help='Number of simulated voxels (default: 100000)')
    parser.add_argument('--grid-density', type=int, default=GRID_DEFAULTS['Ns'],
                        help=f'Grid density for parameter sampling (default: {GRID_DEFAULTS["Ns"]})')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage for batch prediction generation')
    return parser.parse_args()

args = parse_args()

from sweepea import jax_config
jax_backend = jax_config.configure_jax(force_cpu=args.force_cpu)

import sweepea.utilities as utils
from sweepea.visual_stimulus import VisualStimulus
from sweepea.dataloader import load_stimuli
from sweepea.fit_utils import constrain_grids, set_dark_theme
from sweepea.grid_predict import getGridPreds


def main(args,jax_backend):


    # Use a dummy subject to get paths (stimulus is shared)
    params = dict(DEFAULT_PARAMS)
    params['subjID'] = 'JC'
    p, _ = set_paths(params['subjID'], data_format='volumetric')

    # Load stimulus
    bar, _ = load_stimuli(p)
    bar = bar[:, :, 0:201]
    bar = np.flip(bar, axis=0)

    # Create stimulus object
    stimulus = VisualStimulus(
        stim_arr=bar.astype(np.int16),
        viewing_distance=params['viewingDistance'],
        screen_width=params['screenWidth'],
        scale_factor=params['scaleFactor'],
        tr_length=params['tr_length'],
        #dtype=ctypes.c_int16,
    )

    # Build parameter space
    # Currently have code to generate grids more easily, but for now sticking w/ legacy code
    # For consistency w/ older simulations/fitting. Might change that very soon tho.
    Ns = args.grid_density
    x_space = np.concatenate((
        np.linspace(-stimulus.deg_x.max(), stimulus.deg_x.max(), Ns // 2),
        np.geomspace(-stimulus.deg_x.max(), -2 * stimulus.deg_x.max(), Ns // 4),
        np.geomspace(stimulus.deg_x.max(), 2 * stimulus.deg_x.max(), Ns // 4),
    ))
    y_space = np.concatenate((
        np.linspace(-stimulus.deg_y.max(), stimulus.deg_y.max(), Ns // 2),
        np.geomspace(-stimulus.deg_y.max(), -2 * stimulus.deg_y.max(), Ns // 4),
        np.geomspace(stimulus.deg_y.max(), 2 * stimulus.deg_y.max(), Ns // 4),
    ))
    s_space = np.concatenate((
        np.linspace(0.1, 5, 3 * Ns // 4),
        np.geomspace(5, stimulus.deg_x.max(), Ns // 4),
    ))
    n_space = np.linspace(0.01, 1, Ns)

    params_space_orig = list(product(x_space, y_space, s_space, n_space))
    params_space = constrain_grids(params_space_orig, stimulus)

    # Sample random voxels
    nvoxs = args.n_voxels
    params_vox = random.sample(params_space.tolist(), min(nvoxs, len(params_space)))
    baseline_vox = np.random.uniform(0, 10000, len(params_vox))

    # Generate predictions — CPU multiprocessing or GPU batch
    print(f"Generating predictions for {len(params_vox)} simulated voxels "
          f"using {jax_backend.upper()}...") #f"({'GPU' if args.use_gpu else 'CPU'})...")

    # generate hrf
    hrf = utils.double_gamma_hrf(0, params['tr_length'])

    results = getGridPreds(params_vox, stimulus.params, hrf)
   
    # Add noise + linear trend + baseline
    noise_levels = np.random.uniform(0.1, 2.5, results.shape)
    results = (zscore(results, axis=-1, ddof=0)
               + np.random.normal(0, noise_levels, results.shape)
               + np.linspace(0, 1, results.shape[-1])[None, :]
               + baseline_vox[:, None])

    # Save output
    # username = os.getenv("USER")
    # if username == 'nathan':
    #     sim_dir = os.path.join(p['pRF_data'], 'nathan', 'Simulation')
    # else:
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

    # Plot parameter space visualization (Figure 2)
    pv_arr = np.array(params_vox)
    x_vals, y_vals = pv_arr[:, 0], pv_arr[:, 1]
    s_vals, n_vals = pv_arr[:, 2], pv_arr[:, 3]
    rho_vals = np.sqrt(x_vals**2 + y_vals**2)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: RF Centers (x, y) colored by RF size sigma
    sc1 = axs[0, 0].scatter(x_vals, y_vals, c=s_vals, cmap='plasma', s=15, alpha=0.8)
    max_deg = float(stimulus.deg_x.max())
    circle1 = plt.Circle((0, 0), max_deg, color='#00e5ff', fill=False, linestyle='--', linewidth=1.5, label='FOV Max Deg')
    axs[0, 0].add_patch(circle1)
    axs[0, 0].set_aspect('equal', 'box')
    axs[0, 0].set_xlabel('x (deg)')
    axs[0, 0].set_ylabel('y (deg)')
    axs[0, 0].set_title('RF Spatial Centers (x, y) Colored by Size (σ)')
    cbar1 = fig.colorbar(sc1, ax=axs[0, 0])
    cbar1.set_label('RF Size σ (deg)')
    axs[0, 0].grid(True)

    # Panel 2: RF Size (sigma) vs Eccentricity (rho) colored by exponent n
    sc2 = axs[0, 1].scatter(rho_vals, s_vals, c=n_vals, cmap='viridis', s=15, alpha=0.8)
    axs[0, 1].set_xlabel('Eccentricity ρ (deg)')
    axs[0, 1].set_ylabel('RF Size σ (deg)')
    axs[0, 1].set_title('RF Size (σ) vs. Eccentricity (ρ) Colored by Exponent (n)')
    cbar2 = fig.colorbar(sc2, ax=axs[0, 1])
    cbar2.set_label('CSS Exponent n')
    axs[0, 1].grid(True)

    # Panel 3: CSS Exponent (n) Histogram
    axs[1, 0].hist(n_vals, bins=30, color='#ff4081', edgecolor='#121212', alpha=0.85)
    axs[1, 0].set_xlabel('CSS Exponent n')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_title('Distribution of CSS Exponent (n)')
    axs[1, 0].grid(True)

    # Panel 4: Baseline Distribution
    axs[1, 1].hist(baseline_vox, bins=30, color='#76ff03', edgecolor='#121212', alpha=0.85)
    axs[1, 1].set_xlabel('Voxel Baseline')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_title('Distribution of Voxel Baselines')
    axs[1, 1].grid(True)

    plt.suptitle('Simulated pRF Parameter Space Overview', fontsize=16, color='#ffffff', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    param_plot_path = os.path.join(fig_dir, 'simulated_parameter_space.png')
    plt.savefig(param_plot_path, dpi=300)
    plt.close(fig)
    print(f"Parameter space plot saved to {param_plot_path}")


if __name__ == '__main__':
    main(args, jax_backend)

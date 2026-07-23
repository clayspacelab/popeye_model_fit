"""
01_run_pipeline.py — Main pRF fitting pipeline orchestration.

This is the primary entry point for running the CSS pRF model fitting pipeline.
It supports both volumetric (NIfTI, default) and surface (GIFTI) data formats,
with optional GPU acceleration.

Usage:
    # Volumetric (default)
    python 01_run_pipeline.py --subject MAM0606

    # Surface
    python 01_run_pipeline.py --subject MAM0606 --data-format surface

    # With GPU
    python 01_run_pipeline.py --subject MAM0606 --use-gpu

    # Custom grid size, skip final fit
    python 01_run_pipeline.py --subject MAM0606 --grid-size 50 --skip-final-fit
"""

import argparse
import numpy as np
import ctypes
import time
import os

from itertools import product

import popeye.utilities_cclab as utils
from popeye.visual_stimulus import VisualStimulus

# Import pipeline modules
from H01_config import DEFAULT_PARAMS, GRID_DEFAULTS, set_paths
from H02_dataloader import (load_stimuli, load_volumetric_data,
                             load_surface_data, extract_brainmask_voxels,
                             save2gifti, save2nifti)
from H03_fit_utils import print_time, remove_trend, constraint_grids
from H04_grid_predict import getGridPreds
from H05_grid_fit import get_grid_estims
from H06_final_fit import get_final_estims


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='CSS pRF Model Fitting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_run_pipeline.py --subject MAM0606
  python 01_run_pipeline.py --subject MAM0606 --data-format surface --use-gpu
  python 01_run_pipeline.py --subject MAM0606 --grid-size 50 --skip-final-fit
        """
    )
    parser.add_argument('--subject', '-s', required=True,
                        help='Subject ID (e.g., MAM0606)')
    parser.add_argument('--data-format', choices=['volumetric', 'surface'],
                        default='volumetric',
                        help='Data format: volumetric (NIfTI, default) or surface (GIFTI)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Enable GPU acceleration (requires CuPy)')
    parser.add_argument('--grid-size', type=int, default=GRID_DEFAULTS['Ns'],
                        help=f'Grid density parameter Ns (default: {GRID_DEFAULTS["Ns"]})')
    parser.add_argument('--skip-final-fit', action='store_true',
                        help='Skip the gradient-descent final fit step')
    parser.add_argument('--hemisphere', choices=['both', 'left', 'right'],
                        default='both',
                        help='Hemisphere to fit (surface mode only, default: both)')
    return parser.parse_args()


def build_grid_space(stimulus, Ns):
    """
    Construct the 4D parameter search grid and apply constraints.

    Parameters
    ----------
    stimulus : VisualStimulus
        Popeye stimulus object.
    Ns : int
        Grid density parameter.

    Returns
    -------
    grid_space : list
        Constrained grid points as list of (x, y, sigma, n).
    """
    x_grid = np.concatenate((
        np.linspace(-stimulus.deg_x0.max(), stimulus.deg_x0.max(), Ns // 2),
        np.geomspace(-stimulus.deg_x0.max(), -2 * stimulus.deg_x0.max(), Ns // 4),
        np.geomspace(stimulus.deg_x0.max(), 2 * stimulus.deg_x0.max(), Ns // 4),
    ))
    y_grid = np.concatenate((
        np.linspace(-stimulus.deg_y0.max(), stimulus.deg_y0.max(), Ns // 2),
        np.geomspace(-stimulus.deg_y0.max(), -2 * stimulus.deg_y0.max(), Ns // 4),
        np.geomspace(stimulus.deg_y0.max(), 2 * stimulus.deg_y0.max(), Ns // 4),
    ))
    s_grid = np.concatenate((
        np.linspace(0.1, 5, 3 * Ns // 4),
        np.geomspace(5, stimulus.deg_x0.max(), Ns // 4),
    ))
    n_grid = np.asarray(GRID_DEFAULTS['n_grid_values'])

    grid_space_orig = list(product(x_grid, y_grid, s_grid, n_grid))
    grid_space = constraint_grids(grid_space_orig, stimulus)

    return grid_space


def run_volumetric_pipeline(args, p):
    """Run the pRF fitting pipeline on volumetric (NIfTI) data."""

    codeStartTime = time.perf_counter()

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print('Loading volumetric data...')
    scan_data, func_img, metadata = load_volumetric_data(p, method='ss5')
    tr_length = metadata['tr_length']
    nTRs = metadata['nTRs']

    # ── Step 2: Load stimulus ─────────────────────────────────────────────
    print('Loading stimulus...')
    bar, _ = load_stimuli(p)
    bar = bar[:, :, 0:nTRs]  # Trim to match functional data
    bar = np.flip(bar, axis=0)  # Mirror y axis (popeye convention)

    # ── Step 3: Create stimulus object ────────────────────────────────────
    print('Creating stimulus object...')
    stimulus = VisualStimulus(
        bar.astype('int16'),
        DEFAULT_PARAMS['viewingDistance'],
        DEFAULT_PARAMS['screenWidth'],
        DEFAULT_PARAMS['scaleFactor'],
        tr_length,
        DEFAULT_PARAMS['dtype'],
    )

    # ── Step 4: Extract voxels ────────────────────────────────────────────
    print('Extracting voxels...')
    timeseries_data, indices = extract_brainmask_voxels(scan_data)
    print(f'Detrending {len(timeseries_data)} voxels...')
    timeseries_data = remove_trend(timeseries_data, method='all')

    # ── Step 5: Build grid space ──────────────────────────────────────────
    Ns = args.grid_size
    grid_space = build_grid_space(stimulus, Ns)
    print(f'Grid space: {len(grid_space)} points (Ns={Ns})')

    param_width = [
        np.mean(np.diff(np.unique(np.array(grid_space)[:, i])))
        for i in range(4)
    ]

    # ── Step 6: Grid predictions ──────────────────────────────────────────
    gridPath = p['gridfit_path'].replace('.npy', f'_{Ns}.npy')
    tstamp_start = time.perf_counter()

    if os.path.exists(gridPath):
        print(f"Loading grid predictions from {gridPath}")
        grid_preds = np.load(gridPath)
    else:
        print("Generating grid predictions...")
        grid_preds = getGridPreds(grid_space, stimulus, gridPath, nTRs)

    tstamp_gridpred = time.perf_counter()
    print_time(tstamp_start, tstamp_gridpred, 'Grid predictions')

    # ── Step 7: Grid fit ──────────────────────────────────────────────────
    print('Starting grid fit...')
    gFit = np.empty((*scan_data.shape[:3], 9))
    gFit = get_grid_estims(grid_preds, grid_space, timeseries_data, gFit,
                           indices, use_gpu=args.use_gpu)
    tstamp_gridfit = time.perf_counter()
    print_time(tstamp_gridpred, tstamp_gridfit, 'Grid fit')

    # Save grid fit
    gFit_path = os.path.join(p['fitEstimDir'], 'RF_ss5_gFit_popeye.nii.gz')
    save2nifti(gFit, gFit_path, func_img.affine, func_img.header)
    print(f'Grid fit saved to {gFit_path}')

    # ── Step 8: Final fit (optional) ──────────────────────────────────────
    if not args.skip_final_fit:
        print('Starting final fit...')
        fFit = np.empty((*scan_data.shape[:3], 9))
        fFit = get_final_estims(gFit, param_width, timeseries_data, stimulus,
                                fFit, indices, use_gpu=args.use_gpu)
        tstamp_finalfit = time.perf_counter()
        print_time(tstamp_gridfit, tstamp_finalfit, 'Final fit')

        # Save final fit
        fFit_path = os.path.join(p['fitEstimDir'], 'RF_ss5_fFit_popeye.nii.gz')
        save2nifti(fFit, fFit_path, func_img.affine, func_img.header)
        print(f'Final fit saved to {fFit_path}')

    # ── Done ──────────────────────────────────────────────────────────────
    codeEndTime = time.perf_counter()
    print_time(codeStartTime, codeEndTime, 'Total pipeline')


def run_surface_pipeline(args, p, funcFiles):
    """Run the pRF fitting pipeline on surface (GIFTI) data."""

    codeStartTime = time.perf_counter()

    # ── Step 1: Load and average surface data ─────────────────────────────
    print('Loading surface data...')
    leftDataOrig, rightDataOrig, tr_length, nTRs = load_surface_data(p, funcFiles)

    # ── Step 2: Load stimulus ─────────────────────────────────────────────
    print('Loading stimulus...')
    bar, _ = load_stimuli(p)
    bar = np.flip(bar, axis=0)

    # ── Step 3: Create stimulus object ────────────────────────────────────
    print('Creating stimulus object...')
    stimulus = VisualStimulus(
        bar.astype('int16'),
        DEFAULT_PARAMS['viewingDistance'],
        DEFAULT_PARAMS['screenWidth'],
        DEFAULT_PARAMS['scaleFactor'],
        tr_length,
        DEFAULT_PARAMS['dtype'],
    )

    # ── Step 4: Preprocess ────────────────────────────────────────────────
    print('Detrending scan data...')
    leftDataOrig = remove_trend(leftDataOrig, method='all')
    rightDataOrig = remove_trend(rightDataOrig, method='all')

    # ── Step 5: Build grid space ──────────────────────────────────────────
    Ns = args.grid_size
    grid_space = build_grid_space(stimulus, Ns)
    print(f'Grid space: {len(grid_space)} points (Ns={Ns})')

    # ── Step 6: Grid predictions ──────────────────────────────────────────
    gridPath = p['gridfit_path'].replace('.npy', f'_{Ns}.npy')
    tstamp_start = time.perf_counter()

    if os.path.exists(gridPath):
        print(f"Loading grid predictions from {gridPath}")
        grid_preds = np.load(gridPath)
    else:
        print("Generating grid predictions...")
        grid_preds = getGridPreds(grid_space, stimulus, gridPath, nTRs)

    tstamp_gridpred = time.perf_counter()
    print_time(tstamp_start, tstamp_gridpred, 'Grid predictions')

    # ── Step 7–8: Grid fit per hemisphere ─────────────────────────────────
    hemispheres = {}
    if args.hemisphere in ('both', 'left'):
        hemispheres['left'] = leftDataOrig
    if args.hemisphere in ('both', 'right'):
        hemispheres['right'] = rightDataOrig

    for hemi_name, data in hemispheres.items():
        print(f'\n=== Fitting {hemi_name} hemisphere ({data.shape[0]} vertices) ===')

        indices = np.arange(data.shape[0])

        # Grid fit
        print('Starting grid fit...')
        gFit = np.empty((data.shape[0], 9))
        gFit = get_grid_estims(grid_preds, grid_space, data, gFit,
                               indices, use_gpu=args.use_gpu)
        tstamp_gridfit = time.perf_counter()
        print_time(tstamp_gridpred, tstamp_gridfit, f'Grid fit ({hemi_name})')

        # Save grid fit
        gFit_path = os.path.join(p['fitEstimDir'],
                                 f'RF_ss5_gFit_popeye_{hemi_name}.func.gii')
        save2gifti(gFit, fpath=gFit_path, hemisphere=hemi_name)
        print(f'Grid fit saved to {gFit_path}')

        # Final fit (optional)
        if not args.skip_final_fit:
            print('Starting final fit...')
            param_width = [
                np.mean(np.diff(np.unique(np.array(grid_space)[:, i])))
                for i in range(4)
            ]
            fFit = np.empty((data.shape[0], 9))
            fFit = get_final_estims(gFit, param_width, data, stimulus,
                                    fFit, indices, use_gpu=args.use_gpu)
            tstamp_finalfit = time.perf_counter()
            print_time(tstamp_gridfit, tstamp_finalfit, f'Final fit ({hemi_name})')

            fFit_path = os.path.join(p['fitEstimDir'],
                                     f'RF_ss5_fFit_popeye_{hemi_name}.func.gii')
            save2gifti(fFit, fpath=fFit_path, hemisphere=hemi_name)
            print(f'Final fit saved to {fFit_path}')

    codeEndTime = time.perf_counter()
    print_time(codeStartTime, codeEndTime, 'Total pipeline')


def main():
    args = parse_args()

    print(f'=== CSS pRF Model Fitting Pipeline ===')
    print(f'Subject:     {args.subject}')
    print(f'Data format: {args.data_format}')
    print(f'GPU:         {"enabled" if args.use_gpu else "disabled"}')
    print(f'Grid size:   {args.grid_size}')
    print(f'Final fit:   {"skip" if args.skip_final_fit else "enabled"}')
    print()

    # Set up paths
    p, funcFiles = set_paths(args.subject, data_format=args.data_format)

    # Dispatch to appropriate pipeline
    if args.data_format == 'volumetric':
        run_volumetric_pipeline(args, p)
    elif args.data_format == 'surface':
        if funcFiles is None:
            raise RuntimeError("No functional GIFTI files found for surface pipeline.")
        run_surface_pipeline(args, p, funcFiles)


if __name__ == '__main__':
    main()

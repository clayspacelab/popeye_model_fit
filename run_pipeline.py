"""
run_pipeline.py — Main pRF fitting pipeline orchestration.

This is the primary entry point for running the CSS pRF model fitting pipeline.
It supports both volumetric (NIfTI, default) and surface (GIFTI) data formats.
Internally, volumetric data is a single fit target while surface data is one
target per requested hemisphere; both are run through the same fit loop.

Usage:
    # Volumetric (default)
    python run_pipeline.py --subject MAM0606

    # Surface
    python run_pipeline.py --subject MAM0606 --data-format surface

    # Force CPU (otherwise JAX uses GPU automatically if available)
    python run_pipeline.py --subject MAM0606 --force-cpu

    # Custom grid size, skip final fit
    python run_pipeline.py --subject MAM0606 --grid-size 50 --skip-final-fit
"""

import argparse
import numpy as np
import time
import os

from copy import deepcopy
from config import DEFAULT_PARAMS, GRID_DEFAULTS, GRID_PARAMS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='CSS pRF Model Fitting Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --data /path/to/prf_file.nii.gz --stimulus /path/to/stimulus_file.mat
  python run_pipeline.py --data /path/to/prf_file --data-format surface --force-cpu
  python run_pipeline.py --data /path/to/prf_file.nii.gz --grid-size 50 --skip-final-fit
        """
    )
    parser.add_argument('--data', required=True,
                        help='Path to the pRF data file')
    parser.add_argument('--stimulus', required=True,
                        help='Path to the stimulus file')
    parser.add_argument('--data-format', choices=['volume', 'surface'],
                        default='volume',
                        help='Data format: volume (NIfTI, default) or surface (GIFTI)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force JAX to use the CPU even if a GPU is available')
    parser.add_argument('--grid-size', type=int, default=GRID_DEFAULTS['Ns'],
                        help=f'Grid density parameter Ns (default: {GRID_DEFAULTS["Ns"]})')
    parser.add_argument('--skip-final-fit', action='store_true',
                        help='Skip the final fit step')
    # parser.add_argument('--skip-grid-fit', action='store_true',
    #                 help='Skip the grid fit step and load previous prediction')
    # parser.add_argument('--hemisphere', choices=['both', 'left', 'right'],
    #                     default='both',
    #                     help='Hemisphere to fit (surface mode only, default: both)')
    return parser.parse_args()


args = parse_args()

from sweepea import jax_config

jax_backend = jax_config.configure_jax(force_cpu=args.force_cpu)

#JAX-dependent imports must come after the backend is configured
import sweepea.utilities as utils
from sweepea.visual_stimulus import VisualStimulus
from sweepea.dataloader import (set_paths, load_stimuli, load_volumetric_data,
                             load_surface_data, extract_brainmask_voxels,
                             save2gifti, save2nifti, get_gridfit_path)
from sweepea.fit_utils import (print_time, preprocess_signal,
                               constrain_grids,generate_grids,
                               _set_param_minmax,_set_param_gridN)
from sweepea.grid_predict import getGridPreds
from sweepea.grid_fit import get_grid_estims
from sweepea.final_fit import get_final_estims



def build_fit_targets(args, p):
    """
    Load scan data and build the list of fit targets.

    A "fit target" is one independent set of voxels/vertices to run the grid
    and final fit loop over: volumetric data is always a single target,
    surface data is one target per requested hemisphere (one or two).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    p : dict
        Path dictionary from config.set_paths().

    Returns
    -------
    targets : list of dict
        Each has 'name', 'data' (n_units, n_timepoints), 'indices',
        and 'output_shape' (shape for the gFit/fFit output array).
    func_img : nib.Nifti1Image or None
        Present for volumetric data (needed to save NIfTI output), else None.
    """
    if args.data_format == 'volume':
        scan_data, func_img = load_volumetric_data(p)

        print('Extracting voxels...')
        timeseries_data, indices = extract_brainmask_voxels(scan_data)
        print(f'Detrending {len(timeseries_data)} voxels...')
        timeseries_data = preprocess_signal(timeseries_data, detrend_method=p['detrend_method'])

        targets = [{
            'name': 'volume',
            'unit_label': 'voxels',
            'data': timeseries_data,
            'indices': indices,
            'output_shape': (*scan_data.shape[:3], 9),
        }]
        return targets, func_img

    # surface
    elif args.data_format == 'surface':
        hemi_data = load_surface_data(p)

        print('Detrending scan data...')
        for hemi_name, data in hemi_data.items():
            hemi_data[hemi_name] = preprocess_signal(data, detrend_method=p['detrend_method'])

        targets = [{
            'name': hemi_name,
            'unit_label': 'vertices',
            'data': data,
            'indices': np.arange(data.shape[0]),
            'output_shape': (data.shape[0], 9),
        } for hemi_name, data in hemi_data.items()]

        return targets, None
    
    else:
        raise ValueError(f"Unknown data_format '{args.data_format}'. Use 'volume' or 'surface'.")

#this feels like too many inputs but can clean up another time
def save_target_result(target, arr, suffix, p, func_img):
    """Save a grid/final fit result array for one target, format-appropriate."""
    if target['name'] == 'volume':
        fpath = os.path.join(p['fitEstimDir'], f'pRF_{suffix}.nii.gz')
        save2nifti(arr, fpath, func_img.affine, func_img.header)
    else:
        fpath = os.path.join(
            p['fitEstimDir'], f'pRF_{suffix}_hemi-{target["name"][0].upper()}.func.gii'
        )
        save2gifti(arr, fpath=fpath, hemisphere=target['name'])

    print(f'{suffix} saved to {fpath}')


def run_pipeline(args):
    """Run the pRF fitting pipeline on volumetric or surface data."""

    codeStartTime = time.perf_counter()

    # ── Step 0: Load defaults and set paths  ───────────────────────────
    p = deepcopy(DEFAULT_PARAMS)

    # Set up paths (eventually probably a mechanism to set params generally)
    p = set_paths(args.data, data_format=args.data_format, stimulus=args.stimulus, defaults=p)

    # ── Step 1: Load data and build fit targets ───────────────────────────
    print(f'Loading {args.data_format} data...')
    targets, func_img = build_fit_targets(args, p)

    # ── Step 2: Load stimulus, build stimulus object ──────────────────────
    print('Loading stimulus...')
    bar = load_stimuli(p)
    if bar.shape[2] != p['nTRs']:
        raise ValueError(f"Stimulus TRs ({bar.shape[2]}) does not match functional data TRs ({p['nTRs']}).")
    #bar = bar[:, :, 0:p['nTRs']]  # Trim to match functional data (no-op if already matched)
    #bar = np.flip(bar, axis=0)  # Mirror y axis (popeye convention)

    print('Creating stimulus object...')
    stimulus = VisualStimulus(
        bar.astype('int16'),
        p['viewingDistance'],
        p['screenWidth'],
        p['scaleFactor'],
        p['tr_length'],
    )

    # ── Step 3: Build grid space + grid predictions (shared across targets) ─
    Ns = args.grid_size
    grid_params = deepcopy(GRID_PARAMS)
    xy_scale = GRID_DEFAULTS['XY_scale']
    _set_param_gridN(grid_params,Ns)
    _set_param_minmax(grid_params,'x',stimulus.deg_x.min()*xy_scale,stimulus.deg_x.max()*xy_scale)
    _set_param_minmax(grid_params,'y',stimulus.deg_y.min()*xy_scale,stimulus.deg_y.max()*xy_scale)
    _set_param_minmax(grid_params,'s',None,stimulus.deg_x.max())

    grid_space = generate_grids(grid_params,constrain_grids,stimulus)

    print(f'Grid space: {len(grid_space)} points '
          f'(n-grid resolution = {grid_params['n']['num']})')

    tstamp_start = time.perf_counter()
    hrf = utils.double_gamma_hrf(0, p['tr_length'])  # generate hrf. Also used for final fit.

    gridPath = get_gridfit_path(p, grid_space, stimulus.params, hrf,
                                 Ns=Ns, n_res=grid_params['n']['num'])

    if os.path.exists(gridPath):
        print(f"Loading grid predictions from {gridPath}")
        grid_preds = np.load(gridPath)
    else:
        print("Generating grid predictions...")
        grid_preds = getGridPreds(grid_space, stimulus.params, hrf, gridPath)

    tstamp_gridpred = time.perf_counter()
    print_time(tstamp_start, tstamp_gridpred, 'Grid predictions')

    # ── Step 4: Fit each target (volume -> 1 target; surface -> 1 or 2) ───
    for target in targets:
        label = f' ({target["name"]})' if len(targets) > 1 else ''
        print(f'\n=== Fitting {target["name"]} '
              f'({target["data"].shape[0]} {target["unit_label"]}) ===')

        print('Starting grid fit...')
        gFit = np.zeros(target['output_shape'],dtype=np.float32)
        gFit = get_grid_estims(grid_preds, grid_space, target['data'], gFit,
                               target['indices'])
        tstamp_gridfit = time.perf_counter()
        print_time(tstamp_gridpred, tstamp_gridfit, f'Grid fit{label}')

        save_target_result(target, gFit, 'gFit', p, func_img)

        if not args.skip_final_fit:
            print('Starting final fit...')
            fFit = np.zeros(target['output_shape'],dtype=np.float32)
            fFit = get_final_estims(gFit, target['data'], stimulus.params, hrf,
                                    fFit, target['indices'])
            tstamp_finalfit = time.perf_counter()
            print_time(tstamp_gridfit, tstamp_finalfit, f'Final fit{label}')

            save_target_result(target, fFit, 'fFit', p, func_img)

    # ── Done ──────────────────────────────────────────────────────────────
    codeEndTime = time.perf_counter()
    print_time(codeStartTime, codeEndTime, 'Total pipeline')


def main(args, jax_backend):
    print(f'=== CSS pRF Model Fitting Pipeline ===')
    print(f'pRF file path:     {args.data}')
    print(f'Data format: {args.data_format}')
    print(f'JAX backend: {jax_backend.upper()}')
    print(f'Grid size:   {args.grid_size}')
    print(f'Final fit:   {"skip" if args.skip_final_fit else "enabled"}')
    print()

    run_pipeline(args)


if __name__ == '__main__':
    main(args, jax_backend)

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
import importlib.util
import tomli_w

from copy import deepcopy
#import config_example_full


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
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force JAX to use the CPU even if a GPU is available')
    # parser.add_argument('--grid-size', type=int, default=GRID_PARAMS['Ns'],
    #                     help=f'Grid density parameter Ns (default: {GRID_PARAMS["Ns"]})')
    parser.add_argument('--skip-final-fit', action='store_true',
                        help='Skip the final fit step')
    # parser.add_argument('--skip-grid-fit', action='store_true',
    #                 help='Skip the grid fit step and load previous prediction')
    return parser.parse_args()


args = parse_args()

from sweepea import jax_config

jax_backend = jax_config.configure_jax(force_cpu=args.force_cpu)

#JAX-dependent imports must come after the backend is configured
import sweepea.utils as utils
import sweepea.dataloader as dl
import sweepea.config_defaults as config_defaults
from sweepea.visual_stimulus import VisualStimulus
from sweepea.grid_predict import getGridPreds
from sweepea.grid_fit import get_grid_estims
from sweepea.final_fit import get_final_estims



def load_config(config_file=None):
    """Load config.py from an explicit path or the standard search locations."""
    if config_file is None:
        config_file = os.path.join(os.getcwd(), "config.py")

    if os.path.isfile(config_file):
        spec = importlib.util.spec_from_file_location("config", config_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    raise FileNotFoundError(
        "No config.py found: Must include a config.py with stimulus params in working directory or specify with --config"
        )

def set_defaults(configs,defaults):
    """
    Recursively fill missing entries of a nested config dict from defaults.

    Hierarchical analogue of dict.setdefault: any key in `defaults` absent from
    `configs` is added (deep-copied, so later edits don't mutate the defaults);
    where both values are dicts, recurse. Existing user values are never
    overwritten, including None or non-dict values where the default is a dict.
    Modifies `configs` in place and also returns it.
    """
    for key, default_val in defaults.items():
        if key not in configs:
            configs[key] = deepcopy(default_val)
        elif isinstance(configs[key], dict) and isinstance(default_val, dict):
            set_defaults(configs[key], default_val)
    return configs


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
        scan_data, func_img = dl.load_volumetric_data(p)

        print('Extracting voxels...')
        timeseries_data, indices = dl.extract_brainmask_voxels(scan_data)
        print(f'Detrending {len(timeseries_data)} voxels...')
        timeseries_data = utils.preprocess_signal(timeseries_data, detrend_method=p['preproc_params']['detrend_method'])

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
        hemi_data = dl.load_surface_data(p)

        print('Detrending scan data...')
        for hemi_name, data in hemi_data.items():
            hemi_data[hemi_name] = utils.preprocess_signal(data, detrend_method=p['preproc_params']['detrend_method'])

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
def save_target_result(target, arr, suffix, out_dir, func_img):
    """Save a grid/final fit result array for one target, format-appropriate."""
    if target['name'] == 'volume':
        fpath = os.path.join(out_dir, f'pRF_{suffix}.nii.gz')
        ###fpath = os.path.join(out_dir, f'pRF_{suffix}_vistaHRF.nii.gz')
        dl.save2nifti(arr, fpath, func_img.affine, func_img.header)
    else:
        fpath = os.path.join(
            out_dir, f'pRF_{suffix}_hemi-{target["name"][0].upper()}.func.gii'
        )
        dl.save2gifti(arr, fpath=fpath, hemisphere=target['name'])

    print(f'{suffix} saved to {fpath}')


def run_pipeline(args):
    """Run the pRF fitting pipeline on volumetric or surface data."""

    codeStartTime = time.perf_counter()

    # ── Step 0: Load config, set defaults, and set paths  ───────────────────────────
    #this could probably be handled in a cleaner way but fine for now.
    config = load_config(args.config)
    p = {k.lower():v for k,v in config.__dict__.items() if k in config_defaults.CONFIG_FIELDS}
    p_defaults = {k.lower():v for k,v in config_defaults.__dict__.items() if k in config_defaults.CONFIG_FIELDS}
    #p_full = {k.lower():v for k,v in config_example_full.__dict__.items() if k in config_defaults.CONFIG_FIELDS}
    p = set_defaults(p,p_defaults)

    #print(p==p_full)

    # Set up paths (eventually probably a mechanism to set params generally)
    p = dl.set_paths(args.data, data_format=args.data_format, stimulus=args.stimulus, defaults=p)

    # ── Step 1: Load data and build fit targets ───────────────────────────
    print(f'Loading {args.data_format} data...')
    targets, func_img = build_fit_targets(args, p)

    # ── Step 2: Load stimulus, build stimulus object ──────────────────────
    print('Loading stimulus...')
    bar = dl.load_stimuli(p['paths'])
    # if bar.shape[2] != p['nTRs']:
    #     raise ValueError(f"Stimulus TRs ({bar.shape[2]}) does not match functional data TRs ({p['nTRs']}).")
    #bar = bar[:, :, 0:p['nTRs']]  # Trim to match functional data (no-op if already matched)
    #bar = np.flip(bar, axis=0)  # Mirror y axis (popeye convention) [now done w/in load_stimuli]


    #ADD CHECK FOR TYPE OF INPUT/MAKE RESPONSIVE TO DVA INPUT
    print('Creating stimulus object...')
    stimulus = VisualStimulus(
        bar.astype('int16'),
        **p['stimulus_params']
        # p['tr_length'],
        # p['viewingDistance'],
        # p['stimWidth'],
    )

    # ── Step 3: Build grid space + grid predictions (shared across targets) ─
    Ns = p['grid_params']['Ns'] #args.grid_size
    grid_space_params = deepcopy(p['grid_space'])
    xy_scale = p['grid_params']['XY_scale']
    utils._set_param_gridN(grid_space_params,Ns)
    utils._set_param_minmax(grid_space_params,'x',stimulus.deg_x.min()*xy_scale,stimulus.deg_x.max()*xy_scale)
    utils._set_param_minmax(grid_space_params,'y',stimulus.deg_y.min()*xy_scale,stimulus.deg_y.max()*xy_scale)
    utils._set_param_minmax(grid_space_params,'s',None,stimulus.deg_x.max())

    grid_space = utils.generate_grids(grid_space_params,utils.constrain_grids,stimulus)

    print(f'Grid space: {len(grid_space)} points '
          f'(grid size = {Ns}; n-grid resolution = {grid_space_params['n']['num']})')

        #print(f'Grid size:   {args.grid_size}')

    tstamp_start = time.perf_counter()
    hrf = utils.double_gamma_hrf(0, p['stimulus_params']['tr_length'])  # generate hrf. Also used for final fit.
    ###hrf = utils.hrf_two_gammas(p['stimulus_params']['tr_length']) 

    gridPath = dl.get_gridfit_path(p['paths']['stimuli_path'], grid_space, stimulus.params, hrf,
                                 Ns=Ns, n_res=grid_space_params['n']['num'])

    if os.path.exists(gridPath):
        print(f"Loading grid predictions from {gridPath}")
        grid_preds = np.load(gridPath)
    else:
        print("Generating grid predictions...")
        grid_preds = getGridPreds(grid_space, stimulus.params, hrf, gridPath, **p['grid_predict_params'])

    tstamp_gridpred = time.perf_counter()
    utils.print_time(tstamp_start, tstamp_gridpred, 'Grid predictions')

    # ── Step 4: Fit each target (volume -> 1 target; surface -> 1 or 2) ───
    for target in targets:
        label = f' ({target["name"]})' if len(targets) > 1 else ''
        print(f'\n=== Fitting {target["name"]} '
              f'({target["data"].shape[0]} {target["unit_label"]}) ===')

        print('Starting grid fit...')
        gFit = np.zeros(target['output_shape'],dtype=np.float32)
        gFit = get_grid_estims(grid_preds, grid_space, target['data'], gFit,
                               target['indices'], **p['grid_fit_params'])
        tstamp_gridfit = time.perf_counter()
        utils.print_time(tstamp_gridpred, tstamp_gridfit, f'Grid fit{label}')

        save_target_result(target, gFit, 'gFit', p['paths']['fitEstimDir'], func_img)

        if not args.skip_final_fit:
            print('Starting final fit...')
            fFit = np.zeros(target['output_shape'],dtype=np.float32)
            fFit = get_final_estims(gFit, target['data'], stimulus.params, hrf,
                                    fFit, target['indices'], **p['final_fit_params'])
            tstamp_finalfit = time.perf_counter()
            utils.print_time(tstamp_gridfit, tstamp_finalfit, f'Final fit{label}')

            save_target_result(target, fFit, 'fFit', p['paths']['fitEstimDir'], func_img)


    # write out params used for fitting
    with open(os.path.join(p['paths']['fitEstimDir'],'pRF_fitting_params.toml'),'wb') as f:
        tomli_w.dump({k.upper():v for k,v in p.items()},f)

    # ── Done ──────────────────────────────────────────────────────────────
    codeEndTime = time.perf_counter()
    utils.print_time(codeStartTime, codeEndTime, 'Total pipeline')


def main(args, jax_backend):
    print(f'=== CSS pRF Model Fitting Pipeline ===')
    print(f'pRF file path:     {args.data}')
    print(f'Data format: {args.data_format}')
    print(f'JAX backend: {jax_backend.upper()}')
    print(f'Final fit:   {"skip" if args.skip_final_fit else "enabled"}')
    print()

    run_pipeline(args)


if __name__ == '__main__':
    main(args, jax_backend)

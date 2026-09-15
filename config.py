"""
H01_config.py — Central configuration for the pRF fitting pipeline.

Consolidates experiment parameters, host detection, and path generation
from the previously separate dataloader.py and helpersSurface.py modules.

Supports both volumetric (NIfTI, default) and surface (GIFTI) data formats.
"""

import os
import glob
import socket

import numpy as np

# ---------------------------------------------------------------------------
# Default experiment parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    'viewingDistance': 83.5,   # cm (from Zhengang / rsvp_params.txt)
    'screenWidth': 36.2,      # cm
    'scaleFactor': 1,        # currently deprecated and ignored in fitting, but still implemented in VisualStimulus
    'tr_length': 1.3,         # seconds (default, can be overridden by metadata in the NIFTI/GIFTI header, but needed for surface data if not otherwise specified)
    'detrend_method': 'detrend_vista',        # detrending method (default, can be overridden by user)
}

# Grid search defaults
GRID_DEFAULTS = {
    'Ns': 50,                 # grid density
    'XY_scale': 2,         # how many times beyond stimulus to allow x,y positions of prf
    'n_grid_values': [0.25, 0.5, 0.75, 1.0],  # coarse CSS exponent grid (subject pipeline)
    # Finer 10-value CSS exponent grid used by the simulation scripts (S02/S03),
    # since the exponent is the parameter most sensitive to grid resolution.
    'n_grid_values_fine': np.round(np.linspace(0.25, 1.0, 10), 4).tolist(),
}

GRID_PARAMS = {'x': {'space':'lin',
                'num':GRID_DEFAULTS['Ns'],
                },
            'y': {'space':'lin',
                'num':GRID_DEFAULTS['Ns'],
                },
            's': {'space':'linlog',
                'num':GRID_DEFAULTS['Ns'],
                'start':0.1,
                'border':3.0,
                'pctlin':0.6
                },
            'n': {'space':'lin',
                'num':10,
                'start':0.1,
                'stop':1.0
                },
}

# CSS model output field names (9-element tuple per voxel/vertex)
CSS_FIELD_NAMES = (
    'theta',     # polar angle
    'r2',        # variance explained (R²)
    'rho',       # eccentricity
    'sigma',     # RF size
    'n',         # CSS exponent
    'x',         # x-position
    'y',         # y-position
    'beta',      # amplitude (slope)
    'baseline',  # baseline (intercept)
)


# ---------------------------------------------------------------------------
# Host detection
# ---------------------------------------------------------------------------
def detect_host():
    """
    Detect the compute environment based on hostname.

    Returns
    -------
    str
        One of: 'lab_local', 'vader', 'local_mac', 'unknown'
    """
    hostname = socket.gethostname()

    if hostname in ('syndrome', 'zod.psych.nyu.edu', 'zod','doom.psych.nyu.edu'):
        return 'lab_local'
    elif 'vader' in hostname:
        return 'vader'
    elif 'Mrugank' in hostname or 'mrugank' in hostname:
        return 'local_mac'
    else:
        return 'unknown'




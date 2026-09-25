"""
config.py — Central configuration for the pRF fitting pipeline.

"""
#from sweepea.visual_stimulus import dva

# ---------------------------------------------------------------------------
# Stimulus parameters
# ---------------------------------------------------------------------------
STIMULUS_PARAMS = {
    'viewing_distance': 83.5,   # cm (from Zhengang / rsvp_params.txt)
    'stim_width': 36.2,      # cm
    #'stim_dva': dva(36.2,83.5), #24.461,      # dva of stim
    'tr_length': 1.3,         # seconds (default, can be overridden by metadata in the NIFTI/GIFTI header, but needed for surface data if not otherwise specified)
}

# preprocessing of BOLD data
PREPROC_PARAMS = {
    'detrend_method': 'detrend_vista',        # detrending method (default, can be overridden by user)
}

# Grid search defaults (Ns can be overridden at command line)
GRID_PARAMS = {
    'Ns': 50,                 # grid density
    'XY_scale': 2,            # how many times beyond stimulus to allow x,y positions of prf
}

# Grid space construction
GRID_SPACE = {'x': {'space':'lin',
                'num':GRID_PARAMS['Ns'],
                },
            'y': {'space':'lin',
                'num':GRID_PARAMS['Ns'],
                },
            's': {'space':'linlog',
                'num':GRID_PARAMS['Ns'],
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

GRID_PREDICT_PARAMS = {
    'batch_size':2000,
}

GRID_FIT_PARAMS = {
    'batch_size':2000,
}

FINAL_FIT_PARAMS = {
    'batch_size':50000,
}
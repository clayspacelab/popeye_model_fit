#valid config subsections
CONFIG_FIELDS = ['STIMULUS_PARAMS',
                 'PREPROC_PARAMS',
                 'GRID_PARAMS',
                 'GRID_SPACE',
                 'GRID_PREDICT_PARAMS',
                 'GRID_FIT_PARAMS',
                 'FINAL_FIT_PARAMS']


### DEFAULT PARAMETERS ###
# can be overriden wit user-specified config.py file
# user MUST specify STIMULUS_PARAMS in user config.py

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

#Note: batch size params may need to be set by user depending on available GPU resources

#parameters for grid prediction
GRID_PREDICT_PARAMS = {
    'batch_size':2000,
}

#parameters for grid fit
GRID_FIT_PARAMS = {
    'batch_size':GRID_PREDICT_PARAMS['batch_size'],
}

#parameters for final (gradient descent) fit
FINAL_FIT_PARAMS = {
    'batch_size':50000,
}

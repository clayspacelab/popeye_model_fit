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

# CSS model output field names (9-element tuple per voxel/vertex) [currently unused, doesn't belong here]
# CSS_FIELD_NAMES = (
#     'theta',     # polar angle
#     'r2',        # variance explained (R²)
#     'rho',       # eccentricity
#     'sigma',     # RF size
#     'n',         # CSS exponent
#     'x',         # x-position
#     'y',         # y-position
#     'beta',      # amplitude (slope)
#     'baseline',  # baseline (intercept)
# )
import os
import numpy as np
import shutil
from scipy.io import loadmat
import socket

hostname = socket.gethostname()

def set_paths(params):
    subjID = params['subjID']
    p = {}
    if hostname == 'syndrome' or hostname == 'zod.psych.nyu.edu' or hostname == 'zod' or hostname == 'vader':
        # If one of the lab computers with local mount of data server
        p['pRF_data'] = '/d/DATA/data/popeye_pRF/'
        p['orig_data'] = '/d/DATD/datd/pRF_orig/'
    elif hostname == 'log2':
        # Running on HPC
        p['pRF_data'] = '/scratch/mdd9787/popeye_pRF/greene/'
        
    else: # Set paths on local macbook of Mrugank
        p['pRF_data'] = '/Users/mrugankdake/Documents/Clayspace/MRI/popeye_pRF/'
        p['orig_data'] = '/Users/mrugankdake/Documents/Clayspace/MRI/pRF_orig/'

    # else:
        # Set paths on HPC
    p['stimuli_path'] = os.path.join(p['pRF_data'], 'Stimuli')
    p['gridfit_path'] = os.path.join(p['stimuli_path'], 'gridfit.npy')
    # Paths for relevant files from the original data
    p['orig_brainmask'] = os.path.join(p['orig_data'], subjID, 'surfanat_brainmask_hires.nii.gz')
    p['orig_func'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_func.nii.gz')
    p['orig_ss5'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_ss5.nii.gz')
    p['orig_surf'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_surf.nii.gz')
    p['orig_anat'] = os.path.join(p['orig_data'], subjID, 'anat_T1_brain.nii')

    # Paths for the new pRF holder
    p['pRF_brainmask'] = os.path.join(p['pRF_data'], subjID, 'surfanat_brainmask_hires.nii.gz')
    p['pRF_func'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_func.nii.gz')
    p['pRF_ss5'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_ss5.nii.gz')
    p['pRF_surf'] = os.path.join(p['pRF_data'], subjID, 'bar_seq_1_surf.nii.gz')
    p['pRF_anat'] = os.path.join(p['pRF_data'], subjID, 'anat_T1_brain.nii')

    # Figure directory
    if not os.path.exists(os.path.join(p['pRF_data'], subjID)):
        os.mkdir(os.path.join(p['pRF_data'], subjID))
    p['fig_dir'] = os.path.join(p['pRF_data'], subjID, 'figures')
    if not os.path.exists(p['fig_dir']):
        os.mkdir(p['fig_dir'])

    # Copy a folder as a hyperlink
    if hostname == 'syndrome' or hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
        p['orig_anat_dir'] = os.path.join(p['orig_data'], subjID, subjID+'anat')
        p['pRF_anat_dir'] = os.path.join(p['pRF_data'], subjID, subjID+'anat')
        if not os.path.exists(p['pRF_anat_dir']):
            shutil.copy(p['orig_anat_dir'], os.path.join(p['pRF_data'], subjID), follow_symlinks=False)
    return p

def load_stimuli(p):
    '''
    Loads the bar and the params files that should be constant across subjects.
    '''
    bar = loadmat(os.path.join(p['stimuli_path'], 'bar_stimulus_masks_1300ms_images.mat'))['images']
    params = loadmat(os.path.join(p['stimuli_path'], 'bar_stimulus_masks_1300ms_params.mat'))
    return bar, params


def copy_files(p, params):
    subjID = params['subjID']
    if not os.path.exists(os.path.join(p['pRF_data'], subjID)):
        os.mkdir(os.path.join(p['pRF_data'], subjID))
        os.system('cp ' + p['orig_brainmask'] + ' ' + p['pRF_brainmask'])
        os.system('cp ' + p['orig_func'] + ' ' + p['pRF_func'])
        os.system('cp ' + p['orig_ss5'] + ' ' + p['pRF_ss5'])
        os.system('cp ' + p['orig_surf'] + ' ' + p['pRF_surf'])
        os.system('cp ' + p['orig_anat'] + ' ' + p['pRF_anat'])
    else:
        print('Subject folder already exists')

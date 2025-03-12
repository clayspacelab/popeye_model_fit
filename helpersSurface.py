import os
import numpy as np
import shutil
from scipy.io import loadmat
import socket
import nibabel as nib

hostname = socket.gethostname()
print(hostname)

def set_paths_surface(params):
    '''
    Wrapped for the new version of popeye that will deal with surface level data.
    '''
    subjID = params['subjID']
    p = {}
    p['hostname'] = hostname
    if hostname == 'syndrome' or hostname == 'zod.psych.nyu.edu' or hostname == 'zod' or hostname == 'vader':
        # If one of the lab computers with local mount of data server
        p['pRF_data'] = '/d/DATD/datd/popeye_pRF/'
        p['orig_data'] = '/d/DATD/datd/pRF_orig/'
    elif hostname == 'log-1' or hostname == 'log-2' or hostname == 'log-3' or hostname == 'log-4' or 'hpc' in hostname:
        # Running on HPC
        p['pRF_data'] = '/scratch/mdd9787/popeye_pRF_greene/'
    elif 'vader' in hostname:
        # Running on Vader
        p['pRF_data'] = '/clayspace/datd/popeye_pRF/'
        
    else: # Set paths on local macbook of Mrugank
        p['pRF_data'] = '/Users/mrugankdake/Documents/Clayspace/MRI/popeye_pRF/'
        p['orig_data'] = '/Users/mrugankdake/Documents/Clayspace/MRI/pRF_orig/'
        # Paths for relevant files from the original data
        p['orig_brainmask'] = os.path.join(p['orig_data'], subjID, 'surfanat_brainmask_hires.nii.gz')
        p['orig_func'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_func.nii.gz')
        p['orig_ss5'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_ss5.nii.gz')
        p['orig_surf'] = os.path.join(p['orig_data'], subjID, 'RF1', subjID+'_RF1_vista', 'bar_seq_1_surf.nii.gz')
        p['orig_anat'] = os.path.join(p['orig_data'], subjID, 'anat_T1_brain.nii')

    p['stimuli_path'] = os.path.join(p['pRF_data'], 'Stimuli')
    p['gridfit_path_25'] = os.path.join(p['stimuli_path'], 'gridfit_25.npy')
    p['gridfit_path_35'] = os.path.join(p['stimuli_path'], 'gridfit_35.npy')
    p['gridfit_path_50'] = os.path.join(p['stimuli_path'], 'gridfit_50.npy')
    
    # Paths for the new pRF holder
    p['pRF_root'] = os.path.join(p['pRF_data'], 'sub-' + subjID, 'ses-pRF', 'func_smoothed') # Root directory where fMRIprep data is stored
    funcFiles = os.listdir(p['pRF_root'])
    # funcFiles = [f for f in funcFiles if f.endswith('fsnative_mtsmooth-3mm_bold.func.gii')]
    funcFiles = [f for f in funcFiles if f.endswith('fsnative_smoothed_bold.func.gii')]

    # Saving directory
    p['popeyeFitDir'] = os.path.join(p['pRF_data'], 'sub-' + subjID, 'popeyeFit')
    if not os.path.exists(p['popeyeFitDir']):
        os.mkdir(p['popeyeFitDir'])
    # Figure directory
    p['fig_dir'] = os.path.join(p['popeyeFitDir'], 'figs')
    if not os.path.exists(p['fig_dir']):
        os.mkdir(p['fig_dir'])
    # Fit estimates directory
    p['fitEstimDir'] = os.path.join(p['popeyeFitDir'], 'fitEstimates')
    if not os.path.exists(p['fitEstimDir']):
        os.mkdir(p['fitEstimDir'])

    # Copy a folder as a hyperlink
    # if hostname == 'syndrome' or hostname == 'zod.psych.nyu.edu' or hostname == 'zod':
    #     p['orig_anat_dir'] = os.path.join(p['orig_data'], subjID, subjID+'anat')
    #     p['pRF_anat_dir'] = os.path.join(p['pRF_data'], subjID, subjID+'anat')
    #     if not os.path.exists(p['pRF_anat_dir']):
    #         shutil.copy(p['orig_anat_dir'], os.path.join(p['pRF_data'], subjID), follow_symlinks=False)
    return p, funcFiles

def load_stimuli(p):
    '''
    Loads the bar and the params files that should be constant across subjects.
    '''
    bar = loadmat(os.path.join(p['stimuli_path'], 'bar_stimulus_masks_1300ms_images.mat'))['images']
    params = loadmat(os.path.join(p['stimuli_path'], 'bar_stimulus_masks_1300ms_params.mat'))
    return bar, params


def averageRuns(p, funcFiles):
    leftFiles = [f for f in funcFiles if 'L_space' in f]
    rightFiles = [f for f in funcFiles if 'R_space' in f]
    # The number of runs should be the same for both left and right hemispheres
    nRuns = len(leftFiles)

    for run in range(nRuns):
        leftFname = os.path.join(p['pRF_root'], leftFiles[run])
        rightFname = os.path.join(p['pRF_root'], rightFiles[run])

        # Load and extract data
        imgLeft = nib.load(leftFname)
        tempLeft = np.array([x.data for x in imgLeft.darrays])
        tempLeft = np.expand_dims(tempLeft, axis=-1)

        imgRight = nib.load(rightFname)
        tempRight = np.array([x.data for x in imgRight.darrays])
        tempRight = np.expand_dims(tempRight, axis=-1)

        if run == 0:
            leftData = tempLeft
            rightData = tempRight
        else:
            leftData = np.concatenate((leftData, tempLeft), axis=-1)
            rightData = np.concatenate((rightData, tempRight), axis=-1)

    # Average the runs
    leftData = np.mean(leftData, axis=-1).T
    rightData = np.mean(rightData, axis=-1).T

    # Extract some metadata
    # tr_length = int(float(imgLeft.darrays[0].metadata['TimeStep'])) # in ms
    tr_length = 1.3 # in seconds
    nTRs = leftData.shape[1]

    return leftData, rightData, tr_length, nTRs


def save2gifti(fitEstims, fpath):
    # Convert data to float32
    fitEstims = fitEstims.astype(np.float32)
    giiMeta = nib.gifti.GiftiMetaData( {'AnatomicalStructurePrimary': 'CortexLeft', 
                                        'PaletteNormalizationMode': 'NORMALIZATION_SELECTED_MAP_DATA', 
                                        'TimeStep': '1.3', 
                                     } )
    img = nib.gifti.GiftiImage(meta=giiMeta)
    for i in range(fitEstims.shape[1]):
        thisParam = nib.gifti.GiftiDataArray(data = fitEstims[:, i])
        img.add_gifti_data_array(thisParam)
    nib.save(img, fpath)
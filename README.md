# popeye_model_fit

This repository contains tools for fitting population receptive field (pRF) models using the popeye library.

## HPC Setup Instructions

### 1. Connect to HPC

```bash
ssh mdd9787@greene.hpc.nyu.edu
```

### 2. Set up Overlay Image

Navigate to your scratch space and create a directory to store the overlay image:

```bash
cd /scratch/mdd9787/popeye_pRF_greene
mkdir overlay_img
cd overlay_img
```

Copy the overlay with the maximum size (will be needed):

```bash
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
gunzip overlay-50G-10M.ext3.gz
```

### 3. Launch Singularity Container

Choose a singularity image:

```bash
singularity exec --overlay overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash
```

This will launch the singularity image.

### 4. Install Miniconda

Get and install miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
```

### 5. Create Environment Wrapper Script

Create a wrapper script `/ext3/env.sh`:

```bash
nano /ext3/env.sh
```

Add the following content to the script:

```bash
#!/bin/bash

unset -f which

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
```

To activate the environment, run:

```bash
source /ext3/env.sh
```

### 6. Set up Conda Environment

Update conda and create the environment:

```bash
conda update -n base conda -y
conda clean --all --yes
conda install pip -y
conda install ipykernel -y
conda create --name prf_fitter
conda activate prf_fitter
```

Install required packages:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn cython numba tqdm
conda install conda-forge::nibabel nilearn
```

### 7. Install Popeye

For Greene HPC:

```bash
pip install -e /scratch/mdd9787/popeye_pRF_greene/popeye
```

For Vader:

```bash
pip install -e /hyper/toolboxes/popeye
pip install cupy-cuda12x
```

### 8. Final Setup

Rename your overlay:

```bash
mv overlay-50G-10M.ext3 popeye_overlay.ext3
```

## Running the script and progress so far:

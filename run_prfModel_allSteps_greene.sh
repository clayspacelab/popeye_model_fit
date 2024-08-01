#!/bin/bash
#
#SBATCH --job-name=PythonJobe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --output=slurm%j.out
#SBATCH --gres=gpu:1

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# Setting up Conda for Greene
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


###############################################
# set parameters (CHANGE IF NEEDED!!!!!)
###############################################
subj='JC'

###############################################
# run pRF model
###############################################
python /scratch/mdd9787/popeye_model_fit/hpc_popeyeFitter.py ${subj}

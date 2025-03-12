#!/bin/bash
#SBATCH --job-name=MDtestrun                 # The name of the job
#SBATCH --nodes=1                            # Request 1 compute node per job instance
#SBATCH --cpus-per-task=30                    # Reqest 20 CPU per job instance
#SBATCH --mem=100GB                           # Request 50GB of RAM per job instance
#SBATCH --time=24:00:00                      # Request 2 hours per job instance
#SBATCH --output=slurm%j.out
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL

module purge
subj='MAM0606'
root_dir='/scratch/mdd9787/popeye_pRF_greene/popeye_model_fit'
cd $root_dir
chmod 755 activators/activate_conda.bash
activators/activate_conda.bash python hpc_popeyeSurface.py $subj
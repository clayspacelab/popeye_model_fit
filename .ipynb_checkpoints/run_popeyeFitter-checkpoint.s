#!/bin/bash
#SBATCH --job-name=MDtestrun                 # The name of the job
#SBATCH --nodes=1                            # Request 1 compute node per job instance
#SBATCH --cpus-per-task=4                    # Reqest 20 CPU per job instance
#SBATCH --mem=50GB                           # Request 50GB of RAM per job instance
#SBATCH --time=00:10:00                      # Request 2 hours per job instance
#SBATCH --output=slurm%j.out
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1

module purge
subj='JC'
root_dir='/scratch/mdd9787/popeye_pRF_greene/popeye_model_fit'
cd $root_dir
chmod 755 activators/activate_conda.bash
activators/activate_conda.bash python testingCuda.py
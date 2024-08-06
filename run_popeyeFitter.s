#!/bin/bash
#SBATCH --job-name=MDtestrun                 # The name of the job
#SBATCH --nodes=1                            # Request 1 compute node per job instance
#SBATCH --cpus-per-task=1                    # Reqest 1 CPU per job instance
#SBATCH --mem=20GB                           # Request 20GB of RAM per job instance
#SBATCH --time=00:30:00                      # Request 30 mins per job instance
#SBATCH --output=/scratch/mdd9787/
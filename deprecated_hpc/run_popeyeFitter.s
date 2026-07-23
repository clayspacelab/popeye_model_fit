#!/bin/bash
#SBATCH --job-name=MDtestrun                 # The name of the job
#SBATCH --nodes=1                            # Request 1 compute node per job instance
#SBATCH --cpus-per-task=30                    # Reqest 10 CPU per job instance
#SBATCH --mem=32GB                           # Request 8GB of RAM per job instance
#SBATCH --gres=gpu:0
#SBATCH --time=01:30:00                      # Request 0.5 hours per job instance
#SBATCH --output=slurmOutput/slurm%j.out
#SBATCH --mail-user=mrugank.dake@nyu.edu
#SBATCH --mail-type=ALL

mkdir -p slurmOutput

module purge
subj='ZID0704'
root_dir='/scratch/mdd9787/popeye_pRF_greene/popeye_model_fit'
cd $root_dir
chmod 755 activators/activate_conda.bash
activators/activate_conda.bash python hpc_popeyeSurface.py $subj

def monitor_memory():
    """Monitor GPU memory usage"""
    import cupy as cp
    import psutil
    
    process = psutil.Process()
    gpu_memory = cp.cuda.Device().mem_info[0] / 1024**3  # GB
    cpu_memory = process.memory_info().rss / 1024**3  # GB
    
    logging.info(f"GPU Memory Available: {gpu_memory:.2f} GB")
    logging.info(f"CPU Memory Usage: {cpu_memory:.2f} GB")

# Call periodically in your processing loop
monitor_memory()
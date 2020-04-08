#!/bin/bash
#SBATCH --account=cooptimasoot
#SBATCH --time=10-00
#SBATCH --job-name=gaussian
#SBATCH --qos=high
#SBATCH -n 40
#SBATCH -c 18
#SBATCH --no-kill
#SBATCH --output=/scratch/pstjohn/bde/job_output/job_output_filename.%j.out  # %j will be replaced with the job ID

source /home/pstjohn/.bashrc
conda activate /projects/cooptimasoot/pstjohn/envs/rdkit_cpu
ulimit -c 0
cd /scratch/pstjohn

srun python /home/pstjohn/Research/BDE/gaussian_worker.py

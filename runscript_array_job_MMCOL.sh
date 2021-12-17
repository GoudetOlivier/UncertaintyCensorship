#!/bin/bash

#SBATCH --array=0-3
#SBATCH --time=10-23:59:00
#SBATCH -N1
#SBATCH --no-kill
#SBATCH --error=slurm-err-%j.out
#SBATCH --output=slurm-o-%j.out
#SBATCH --cpus-per-task=10	
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000




python main_MNAR.py $SLURM_ARRAY_TASK_ID

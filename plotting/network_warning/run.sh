#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=warning         # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-1               # job array id

#SBATCH --mem-per-cpu=24G		       # Memory Request (per CPU; can use on GLIC)
#SBATCH --begin=2023-10-20T09:00:00# job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/plotting/network_warning/logs/out.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/plotting/network_warning/logs/err.txt 		# Standard Error Log File (for Job Arrays)


source /home/qizhou/miniforge3/bin/activate
conda activate seismic

# Run your Python script using srun with the parameters
srun python /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/plotting/network_warning/warning_strategy.py

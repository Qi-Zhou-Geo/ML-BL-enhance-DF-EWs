#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=create_label    # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --mem-per-cpu=16G		       # Memory Request (per CPU; can use on GLIC)

#SBATCH --output /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/create_labels/out.txt
#SBATCH --error  /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/create_labels/err.txt


source /home/qizhou/miniforge3/bin/activate
conda activate seismic


srun python /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/create_labels/create_df_labels.py



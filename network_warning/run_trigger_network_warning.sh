#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=network         # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-36               # job array id

#SBATCH --mem-per-cpu=4G		       # Memory Request (per CPU; can use on GLIC)
#SBATCH --begin=2023-10-20T09:00:00# job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output /home/qizhou/3paper/2AGU/3modelMigration/logs/out_%A_%a_%x.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/3paper/2AGU/3modelMigration/logs/err_%A_%a_%x.txt 		# Standard Error Log File (for Job Arrays)


source /home/qizhou/miniforge3/bin/activate
conda activate ml


# Define arrays for parameter1, parameter2, and parameter3
parameter1=("RF" "XGB" "LSTM")
parameter2=("A" "B" "C" "D")
parameter3=(2013 2014 2020)


# Calculate the indices for the input parameter combination
parameter1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameter3[@]} * ${#parameter2[@]} ) % ${#parameter1[@]} + 1 ))
parameter2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameter3[@]} % ${#parameter2[@]} + 1 ))
parameter3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameter3[@]} + 1 ))

# Get the current parameter values
current_parameter1=${parameter1[$parameter1_idx - 1]}
current_parameter2=${parameter2[$parameter2_idx - 1]}
current_parameter3=${parameter3[$parameter3_idx - 1]}

# Print the current combination
echo "input station: $current_parameter1, input feature: $current_parameter2, input model: $current_parameter3"

srun python /home/qizhou/3paper/2AGU/3modelMigration/3NetworkWarning.py --RF_or_XGB "$current_parameter1" --bl_or_rf "$current_parameter2" --data_year "$current_parameter3"


#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=dual_ensemble   # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-6               # job array id

#SBATCH --mem-per-cpu=64G		       # Memory Request (per CPU; can use on GLIC)
#SBATCH --begin=2023-10-20T09:00:00# job start time, if it later than NOW, job will be run immediatly.

#SBATCH --chdir=/home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/functions # set working dir
#SBATCH --output=model_dual_test/logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=model_dual_test/logs/err_%A_%a_%x.txt   # Standard Error Log File


source /home/qizhou/miniforge3/bin/activate
conda activate ml


parameters1=("Random_Forest" "XGBoost") # input model
parameters2=("A" "B" "C") # input features
parameters3=("E19A") # input station


# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}


# Run your Python script using srun with the parameters
srun python tree_ensemble_main_dual.py \
     --model_type "$current_parameters1" \
     --feature_type "$current_parameters2" \
     --input_component "CHZ" \
     --ref_station "ILL12" \
     --ref_component "EHZ" \
     --input_seis_network "1A" \
     --input_station "$current_parameters3" \
     --input_data_year 2021

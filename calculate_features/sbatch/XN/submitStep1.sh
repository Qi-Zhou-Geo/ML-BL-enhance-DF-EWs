#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=step1           # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-232             # job array id, !!!! all combination of parameters1-3
#SBATCH --mem-per-cpu=32G		       # Memory Request (per CPU; can use on GLIC)

#SBATCH --chdir=/home/qizhou/3paper/0seismic_feature/sbatch/XN/logs # set working dir
#SBATCH --output=step1/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=step1/err_%A_%a_%x.txt   # Standard Error Log File


source /home/qizhou/miniforge3/bin/activate
conda activate seismic


# Define arrays for parameters1, parameters2, and parameters3
parameters1=(2016)
parameters2=("NEP08" "NEP07" "NEP06" "NEP10" "NEP16" "NEP05" "NEP04" "NEP13") # 8 station from upstream to downstream
parameters3=($(seq 171 199)) # 29 = 199 - 171 + 1


# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}

# Print the current combination
echo "Year: $current_parameters1, Station: $current_parameters2, Julday $current_parameters3"

srun python /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/1cal_TypeA_TypeB.py \
    --seismic_network "XN" \
    --input_year "$current_parameters1" \
    --input_station "$current_parameters2" \
    --input_component "HHZ" \
    --input_window_size 60 \
    --id "$current_parameters3"

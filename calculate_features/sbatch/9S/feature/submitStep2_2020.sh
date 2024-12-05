#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=step2           # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-101               # job array id

#SBATCH --mem-per-cpu=16G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --chdir=/home/qizhou/3paper/0seismic_feature/sbatch/9S/feature/logs # set working dir
#SBATCH --output=step2/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=step2/err_%A_%a_%x.txt   # Standard Error Log File

source /home/qizhou/miniforge3/bin/activate
conda activate seismic


# Define arrays for parameters1, parameters2, and parameters3
parameters1=(2020)
parameters2=("EHZ")
parameters3=($(seq 150 250)) # 101 = 250 - 150 + 1

parameters4=("ILL18" "ILL12" "ILL13")


# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}

# Print the current combination
echo "Year: $current_parameters1, Component: $current_parameters2, Julday: $current_parameters3, Station list: ${parameters4[@]}"

# Run your Python script using srun with the parameters
srun python /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/2cal_TypeB_network.py \
     --seismic_network "9S" \
     --input_year "$current_parameters1" \
     --station_list "${parameters4[@]}" \
     --input_component "$current_parameters2" \
     --input_window_size 60 \
     --id "$current_parameters3"

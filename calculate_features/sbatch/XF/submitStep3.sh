#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=step3           # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-1                # job array id

#SBATCH --mem-per-cpu=8G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --chdir=/home/qizhou/3paper/0seismic_feature/sbatch/XF/logs # set working dir
#SBATCH --output=step3/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=step3/err_%A_%a_%x.txt   # Standard Error Log File


source /home/qizhou/miniforge3/bin/activate
conda activate seismic

# Define arrays for parameters1, parameters2, and parameters3
parameters1=(2003)
parameters2=("H0390")
parameters3=("BHZ")

# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}

# Print the current combination
echo "Year: $current_parameters1, Station: $current_parameters2, Component: $current_parameters3"

# Run your Python script using srun with the parameters
srun python /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/3merge_single_julday.py \
    --seismic_network "XF" \
    --input_year "$current_parameters1" \
    --input_station "$current_parameters2" \
    --input_component "$current_parameters3" \
    --id1 221 \
    --id2 229
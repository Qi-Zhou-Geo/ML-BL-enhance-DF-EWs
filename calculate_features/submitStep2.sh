#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=step2           # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-46               # job array id

#SBATCH --mem-per-cpu=16G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --output /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/logs/step2/out_%A_%a_%x.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/logs/step2/err_%A_%a_%x.txt 		# Standard Error Log File (for Job Arrays)

source /home/qizhou/miniforge3/bin/activate
conda activate seismic


# Define arrays for parameter1 and parameter2
parameter1=($(seq 138 183))
parameter1_idx=$((SLURM_ARRAY_TASK_ID - 1))
current_parameter1="${parameter1[$parameter1_idx]}"

parameter2=("ILL08" "ILL02" "ILL03")

# Print the current combination
echo "julday: $current_parameter1, station_list: $parameter2"

# Run your Python script using srun with the parameters
srun python /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/2cal_TypeB_network.py \
     --input_year 2017 \
     --input_component "EHZ" \
     --input_window_size 60 \
     --id "$current_parameter1" \
     --station_list "${parameter2[@]}"
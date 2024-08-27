#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=step3           # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-3                # job array id

#SBATCH --mem-per-cpu=8G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --output /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/logs/step3/out_%A_%a_%x.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/logs/step3/err_%A_%a_%x.txt 		# Standard Error Log File (for Job Arrays)


source /home/qizhou/miniforge3/bin/activate
conda activate seismic

# Define arrays for parameter1 and parameter2
parameter1=("ILL08" "ILL02" "ILL03")
parameter1_idx=$((SLURM_ARRAY_TASK_ID - 1))
current_parameter1="${parameter1[$parameter1_idx]}"

# Print the current combination
echo "station: $current_parameter1"


srun python /home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs/calculate_features/3merge_single_julday.py \
    --input_year 2017 \
    --input_station "$current_parameter1" \
    --input_component "EHZ" \
    --id1 138 \
    --id2 183
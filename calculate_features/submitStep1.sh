#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=1step1          # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-36               # job array id 

#SBATCH --mem-per-cpu=16G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --output /home/qizhou/1projects/dataForML/out60/logs/step1/2017-2020/out_%A_%a_%x.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/1projects/dataForML/out60/logs/step1/2017-2020/err_%A_%a_%x.txt 		# Standard Error Log File (for Job Arrays)


source /home/qizhou/miniforge3/bin/activate
conda activate seismic


# Define arrays for parameters1, parameters2, and parameters3
parameters1=("2017" "2018" "2019" "2020")
parameters2=("ILL18" "ILL12" "ILL13")
parameters3=("EHE" "EHN" "EHZ")


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

srun python /home/qizhou/1projects/dataForML/out60/1cal_allBL_RF.py --input_year "$current_parameters1" --input_station "$current_parameters2" --input_component "$current_parameters3" --input_window_size 60



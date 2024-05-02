#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=step2           # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-12               # job array id 

#SBATCH --mem-per-cpu=16G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --output /home/qizhou/1projects/dataForML/out60/logs/step2/out_%A_%a_%x.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/1projects/dataForML/out60/logs/step2/err_%A_%a_%x.txt 		# Standard Error Log File (for Job Arrays)

# Define arrays for parameters1, stations, and parameters2
parameters1=("2017" "2018" "2019" "2020")
parameters2=("EHE" "EHN" "EHZ")

# Calculate indices for each array
parameters1_idx=$(( (($SLURM_ARRAY_TASK_ID - 1) / ${#parameters2[@]}) % ${#parameters1[@]} ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters2[@]} ))


# Use the calculated indices to get the corresponding values
current_parameter1="${parameters1[$parameters1_idx]}"
current_parameter2="${parameters2[$parameters2_idx]}"


# Print out the parameters for reference
echo "Processing year $current_parameter1, and component $current_parameter2"


source /home/qizhou/miniforge3/bin/activate
conda activate seismic

# Run your Python script using srun with the parameters
srun python /home/qizhou/1projects/dataForML/out60/2cal_allBL_RF_network.py --year "$current_parameter1" --input_component "$current_parameter2"

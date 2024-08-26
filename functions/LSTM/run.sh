#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=lstm            # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-12               # job array id

#SBATCH --mem-per-cpu=64G              # Memory Request (per CPU; can use on GLIC)
#SBATCH --gres=gpu:A100:1              # load GPU A100 could be replace by A40/A30, 509-510 nodes has 4_A100_80G
#SBATCH --reservation=GPU              # reserve the GPU


#SBATCH --begin=2023-10-20T09:00:00# job start time, if it later than NOW, job will be run immediatly.

#SBATCH --output /home/qizhou/3paper/2AGU/2LSTM/out60_reTest/logs/out_%A_%a_%x.txt         # Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/3paper/2AGU/2LSTM/out60_reTest/logs/err_%A_%a_%x.txt         # Standard Error Log File (for Job Arrays)


# add untested gpu-software stack
module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake

# load environment
source /home/qizhou/miniforge3/bin/activate
conda activate ml


# Define arrays for parameter1, parameter2, and parameter3
# please change [# input station, # input features, # sequence_length]
parameter1=("ILL18" "ILL12" "ILL13") # input station
parameter2=("A" "B" "C" "D") # input features
parameter3=(64) # sequence_length, # batch_size was fixed as 16


# Calculate the indices for the input parameter combination
parameter1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameter3[@]} * ${#parameter2[@]} ) % ${#parameter1[@]} + 1 ))
parameter2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameter3[@]} % ${#parameter2[@]} + 1 ))
parameter3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameter3[@]} + 1 ))

# Get the current parameter values
current_parameter1=${parameter1[$parameter1_idx - 1]}
current_parameter2=${parameter2[$parameter2_idx - 1]}
current_parameter3=${parameter3[$parameter3_idx - 1]}

# Print the current combination
echo "input station: $current_parameter1, input feature: $current_parameter2, input sequence_length: $current_parameter3"

srun --gres=gpu:A100:1 python /home/qizhou/3paper/2AGU/2LSTM/out60_reTest/2RNN_LSTM.py \
     --input_station  "$current_parameter1" \
     --feature_type   "$current_parameter2" \
     --sequence_length "$current_parameter3" \
     --batch_size 16


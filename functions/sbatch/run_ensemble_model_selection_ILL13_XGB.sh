#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=ensemble        # job name, "Qi_run"
#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-79               # job array id (since you have 79 indexes)

#SBATCH --mem-per-cpu=64G		       # Memory Request (per CPU; can use on GLIC)
#SBATCH --begin=2023-10-20T09:00:00# job start time, if it later than NOW, job will be run immediatly.
#SBATCH --output /home/kshitkar/ML-BL-enhance-DF-EWs/output/logs/out_%A_%a_%x.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/kshitkar/ML-BL-enhance-DF-EWs/output/logs/err_%A_%a_%x.txt 		# Standard Error Log File (for Job Arrays)

# Activate Environment
source /home/kshitkar/miniforge3/bin/activate
conda activate ml_env_2

parameters1=("ILL13") # input station
parameters2=("XGBoost") # input model
parameters3=("C") # input features

# List of least important feature indexes (in decreasing order)
# least_important_indexes=(73 72 18 63 62 76 13 74 77 75 53 45 66 59 16 3 10 32 64 4 67 5 69 79 61 40 8 50 46 65 1 6 11 36 7 21 51 39 71 42 17 9 52 31 12 26 25 43 2 22 33 0 58 47 27 78 57 55 34 56 24 23 35 47 25 55 14 41 60 56 31 34 29 28 26 27)
least_important_indexes=(
1
53
50
76
5
11
2
44
48
63
45
13
54
75
27
74
32
77
33
59
66
64
16
15
0
62
79
6
58
68
20
61
43
51
17
65
49
42
29
25
19
72
8
46
9
35
37
52
3
38
34
47
4
10
31
36
60
73
7
69
22
40
12
56
55
57
26
41
70
78
21
67
71
24
39
30
28
18
23
14
)

# Check if SLURM_ARRAY_TASK_ID is set and greater than 0
if [[ -z "$SLURM_ARRAY_TASK_ID" || "$SLURM_ARRAY_TASK_ID" -le 0 ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID must be set and greater than 0."
    exit 1
fi

# Dynamically create exclude_list based on SLURM_ARRAY_TASK_ID
# Extract a slice of least_important_indexes up to SLURM_ARRAY_TASK_ID
exclude_list=(${least_important_indexes[@]:0:$SLURM_ARRAY_TASK_ID})

# Print the dynamically created exclude_list
echo "Exclude list for task $SLURM_ARRAY_TASK_ID: ${exclude_list[@]}"

# Initialize parameters4 array with integers from 0 to 79, excluding the items in exclude_list
parameters4=()
for i in {0..79}; do
    if [[ ! " ${exclude_list[@]} " =~ " ${i} " ]]; then
        parameters4+=($i)
    fi
done

# Print the final parameters4 array (integers 0-79, excluding exclude_list items)
echo "Final parameters4: ${parameters4[@]}"

# Convert parameters4 to a comma-separated string
parameters4_string=$(IFS=,; echo "${parameters4[*]}")

# Calculate the indices for the current combination
parameters1_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ( ${#parameters2[@]} * ${#parameters3[@]} ) % ${#parameters1[@]} + 1 ))
parameters2_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) / ${#parameters3[@]} % ${#parameters2[@]} + 1 ))
parameters3_idx=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#parameters3[@]} + 1 ))

# Get the current parameter values
current_parameters1=${parameters1[$parameters1_idx - 1]}
current_parameters2=${parameters2[$parameters2_idx - 1]}
current_parameters3=${parameters3[$parameters3_idx - 1]}

# Run your Python script using srun with the parameters
srun python /home/kshitkar/ML-BL-enhance-DF-EWs/functions/temp_tree_ensemble_main.py \
     --input_station "$current_parameters1" \
     --model_type "$current_parameters2" \
     --feature_type "$current_parameters3" \
     --input_component "EHZ" \
     --indexes "$parameters4_string"

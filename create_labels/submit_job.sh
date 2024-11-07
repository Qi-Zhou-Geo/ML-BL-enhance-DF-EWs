#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=create_label    # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-1                # job array id

#SBATCH --mem-per-cpu=16G		       # Memory Request (per CPU; can use on GLIC)

#SBATCH --chdir=/home/qizhou/3paper/1seismic_label/sbatch # set working dir
#SBATCH --output=out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=err_%A_%a_%x.txt   # Standard Error Log File


source /home/qizhou/miniforge3/bin/activate
conda activate seismic


# Run your Python script using srun with the parameters
srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2017 \
     --input_station "ILL02" \
     --input_component "EHZ"

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2018 \
     --input_station "ILL12" \
     --input_component "EHZ"

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2019 \
     --input_station "ILL12" \
     --input_component "EHZ"

srun python create_debris_flow_labels.py \
     --seismic_network "9S" \
     --input_year 2020 \
     --input_station "ILL12" \
     --input_component "EHZ"

#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=data_prepare    # job name, "Qi_run"

#SBATCH --chdir=/home/qizhou/3paper/0seismic_feature/sbatch/9S/3rd/logs # set working dir
#SBATCH --output=out_data_prepare.txt  # Standard Output Log File
#SBATCH --error=err_data_prepare.txt   # Standard Error Log File

# Define paths to your job scripts
job1="/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature/sbatch/9S/3rd/submit2017.sh"
job2="/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature/sbatch/9S/3rd/submit2018_2019.sh"
job3="/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature/sbatch/9S/3rd/submit2020.sh"
job4="/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature/sbatch/9S/3rd/submit2022.sh"

# Submit the jobs in parallel
sbatch $job1 &
sbatch $job2 &
sbatch $job3 &
sbatch $job4 &

# Wait for all jobs to finish
wait

echo "All jobs have been submitted and are running in parallel."

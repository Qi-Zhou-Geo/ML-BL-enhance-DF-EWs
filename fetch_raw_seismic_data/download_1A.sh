#!/bin/bash

# Base directory for downloaded data
base_dir="/Users/qizhou/Downloads/Museum_Fire"


# Variables for the station and network
net="1A"
year="2021"


# List of stations
station_list=("COCB" "E19A")
# List of channel_list
channel_list=("CHZ" "CHE" "CHN")
# List of Julian days
julian_day_list=($(seq 174 239)) # [174, 175,..., 239]



# Loop over each Julian day
for sta in "${station_list[@]}"
do
  for julian_day in "${julian_day_list[@]}"
  do
    # # Convert Julian day to the corresponding date macOS
    start_time=$(date -jf "%Y%j" "$year$julian_day" +%Y-%m-%dT00:00:00.000)
    end_time=$(date -jf "%Y%j" "$year$((julian_day + 1))" +%Y-%m-%dT00:00:00.000)

    # Convert Julian day to the corresponding date in Linux
    #start_time=$(date -d "$year-01-01 +$((julian_day - 1)) days" +%Y-%m-%dT00:00:00.000)
    #end_time=$(date -d "$year-01-01 +$julian_day days" +%Y-%m-%dT00:00:00.000)

    # Loop over each channel
    for cha in "${channel_list[@]}"
    do
        # Create the directory structure for each channel and day
        output_dir="${base_dir}/${year}/${sta}/${cha}"
        mkdir -p "$output_dir"

        # Download the data using curl
        curl -o "${output_dir}/${net}.${sta}.${cha}.${year}.${julian_day}.mseed" \
             "http://service.iris.edu/fdsnws/dataselect/1/query?net=${net}&sta=${sta}&loc=00&cha=${cha}&start=${start_time}&end=${end_time}"

        echo "Downloaded ${cha} data for Julian day ${julian_day} to ${output_dir}/${net}.${sta}.${cha}.${year}.${julian_day}.mseed"
    done
  done
done

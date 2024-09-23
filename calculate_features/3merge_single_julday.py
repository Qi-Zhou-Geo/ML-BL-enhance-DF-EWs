#!/usr/bin/python
# -*- coding: UTF-8 -*-


#__modification time__ = 2024-05-27
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission


import os
import argparse
import numpy as np
import pandas as pd

# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir


def merge_files(input_file_dir, input_files, output_file):

    with open(output_file, 'w') as outfile:
        header_written = False

        for filename in input_files:
            file_path = os.path.join(input_file_dir, filename)

            # Open each file for reading
            with open(file_path, 'r') as infile:
                # Read the header
                header = infile.readline()

                # Write the header only once
                if not header_written:
                    outfile.write(header)
                    header_written = True

                # Write the remaining lines (skip the header)
                for line in infile:
                    outfile.write(line)


def main(input_year, input_station, input_component, id1, id2):

    # Type A
    folder_path = f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}/{input_station}/{input_component}/txt/"
    input_file_dir = folder_path
    output_file = f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}_{input_station}_{input_component}_all_A.txt"
    input_files = [f"{input_year}_{input_station}_{input_component}_{i}_A.txt" for i in range(id1, id2 + 1)]
    merge_files(input_file_dir, input_files, output_file)

    # Type B
    folder_path = f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}/{input_station}/{input_component}/txt/"
    input_file_dir = folder_path
    output_file = f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}_{input_station}_{input_component}_all_B.txt"
    input_files = [f"{input_year}_{input_station}_{input_component}_{i}_B.txt" for i in range(id1, id2 + 1)]
    merge_files(input_file_dir, input_files, output_file)

    # Type B network
    folder_path = f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}/network/{input_component}"
    input_file_dir = folder_path
    output_file = f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}_{input_component}_all_network.txt"
    input_files = [f"{input_year}_{input_component}_{i}_net.txt" for i in range(id1, id2 + 1)]
    merge_files(input_file_dir, input_files, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_year", type=int, default=2020, help="check the year")
    parser.add_argument("--input_station", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_component", type=str, default="EHZ", help="check the input_component")
    parser.add_argument("--id1", type=int, default=1, help="check the julday_id1")
    parser.add_argument("--id2", type=int, default=365, help="check the julday_id1")

    args = parser.parse_args()
    main(args.input_year, args.input_station, args.input_component, args.id1, args.id2)

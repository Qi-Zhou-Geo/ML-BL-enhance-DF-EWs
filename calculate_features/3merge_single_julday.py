#!/usr/bin/python
# -*- coding: UTF-8 -*-


#__modification time__ = 2024-05-27
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission


import os
import platform
import sys
import argparse

import numpy as np
import pandas as pd

# <editor-fold desc="define the parent directory">
if platform.system() == 'Darwin':
    parent_dir = "/Users/qizhou/#python/#GitHub_saved/ML-BL-enhance-DF-EWs"
elif platform.system() == 'Linux':
    parent_dir = "/home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs"
else:
    print(f"check the parent_dir for platform.system() == {platform.system()}")
# add the parent_dir to the sys
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
else:
    pass
# </editor-fold>


# import the custom functions
from config.config_dir import CONFIG_dir, path_mapping


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


def main(seismic_network, input_year, input_station, input_component, id1, id2):
    folder_path = f"{CONFIG_dir['feature_output_dir']}/{path_mapping(seismic_network)}/{input_year}/{input_station}/{input_component}"
    CONFIG_dir['txt_dir'] = folder_path

    # Type A
    input_file_dir = CONFIG_dir['txt_dir']
    output_file = f"{CONFIG_dir['txt_dir']}/{input_year}_{input_station}_{input_component}_all_A.txt"
    input_files = [f"{input_year}_{input_station}_{input_component}_{str(i).zfill(3)}_A.txt" for i in range(id1, id2 + 1)]
    merge_files(input_file_dir, input_files, output_file)

    # Type B
    input_file_dir = CONFIG_dir['txt_dir']
    output_file = f"{CONFIG_dir['txt_dir']}/{input_year}_{input_station}_{input_component}_all_B.txt"
    input_files = [f"{input_year}_{input_station}_{input_component}_{str(i).zfill(3)}_B.txt" for i in range(id1, id2 + 1)]
    merge_files(input_file_dir, input_files, output_file)

    # Type B network
    input_file_dir = f"{CONFIG_dir['feature_output_dir']}/{path_mapping(seismic_network)}/{input_year}/{input_component}_net"
    output_file = f"{input_file_dir}/{input_year}_{input_component}_all_network.txt"
    input_files = [f"{input_year}_{input_component}_{str(i).zfill(3)}_net.txt" for i in range(id1, id2 + 1)]
    merge_files(input_file_dir, input_files, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seismic_network", type=str, default="9S", help="check the year")
    parser.add_argument("--input_year", type=int, default=2020, help="check the year")
    parser.add_argument("--input_station", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_component", type=str, default="EHZ", help="check the input_component")
    parser.add_argument("--id1", type=int, default=1, help="check the julday_id1")
    parser.add_argument("--id2", type=int, default=365, help="check the julday_id1")

    args = parser.parse_args()
    main(args.seismic_network, args.input_year, args.input_station, args.input_component, args.id1, args.id2)

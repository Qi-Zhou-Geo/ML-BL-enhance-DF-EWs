#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import argparse

import pandas as pd
import numpy as np
from obspy.core import UTCDateTime # default is UTC+0 time zone


def create_label(input_year, input_station, input_component):

    if input_station == "ILL08" or input_station == "ILL18":
        usecols = [2, 3]
    elif input_station == "ILL02" or input_station == "ILL12":
        usecols = [5, 6]
    elif input_station == "ILL03" or input_station == "ILL13":
        usecols = [7, 8]
    else:
        print(f"check the input station: {input_station}")

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    folder_path = f"{parent_dir}/data/seismic_feature/"
    df0 = pd.read_csv(f"{folder_path}{input_year}_{input_station}_{input_component}_all_A.txt", header=0)

    folder_path = f"{parent_dir}/create_labels/"
    df1 = pd.read_csv(f"{folder_path}{input_year}_DF.txt", header=0, usecols=usecols)

    df_label = df0.iloc[:, :2]
    df_label["label_0nonDF_1DF"] = 0 # non-debris flow event with label 0
    date = np.array(df_label.iloc[:, 0])

    for step in range(len(df1)):
        id1 = df1.iloc[step, 0]
        id2 = df1.iloc[step, 1]

        id1 = np.where(date == id1)[0][0]
        id2 = np.where(date == id2)[0][0]

        df_label.iloc[id1:id2, -1] = np.full(id2-id1, 1) # debris flow event with label 1

    df_label.to_csv(f"{parent_dir}/data/event_label/{input_year}_{input_station}_{input_component}_observed_label.txt",index=False)
    print(f"{input_year}_{input_station}_{input_component}, done")

def main():

    for input_year in [2017]:
        for input_station in ["ILL08", "ILL02", "ILL03"]:
            for input_component in ["EHZ"]:
                create_label(input_year, input_station, input_component)

    for input_year in [2018, 2019, 2020]:
        for input_station in ["ILL18", "ILL12", "ILL13"]:
            for input_component in ["EHZ"]:
                create_label(input_year, input_station, input_component)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')
    args = parser.parse_args()

    main()


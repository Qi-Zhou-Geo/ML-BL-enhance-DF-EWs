#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys
import argparse

import pandas as pd
import numpy as np
from obspy.core import UTCDateTime # default is UTC+0 time zone
from datetime import datetime


# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# internal functions
from config.config_dir import CONFIG_dir


def write_down_warning(model_type, feature_type, input_component, warning_type, record):
    f = open(f"{CONFIG_dir['output_dir']}/dual_test_9S_warning/{model_type}_{feature_type}_{input_component}_{warning_type}.txt", 'a')
    f.write(str(record) + "\n")
    f.close()


def merge_multi_detection(input_station_list, model_type, feature_type, input_component):

    df0 = pd.read_csv(f"{CONFIG_dir['output_dir']}/dual_test_9S/"
                      f"{input_station_list[0]}_{model_type}_{feature_type}_{input_component}_testing_output.txt", header=0)

    df1 = pd.read_csv(f"{CONFIG_dir['output_dir']}/dual_test_9S/"
                      f"{input_station_list[1]}_{model_type}_{feature_type}_{input_component}_testing_output.txt", header=0)

    df2 = pd.read_csv(f"{CONFIG_dir['output_dir']}/dual_test_9S/"
                      f"{input_station_list[2]}_{model_type}_{feature_type}_{input_component}_testing_output.txt", header=0)

    assert len(df0) == len(df1) or len(df0) == len(df2), f"check the data length for " \
                                                         f"{input_station_list, model_type, feature_type, input_component}"

    df = pd.concat([df0, df1.iloc[:, 1:], df2.iloc[:, 1:]], axis=1)

    return df


def warning_controller(pro_arr, pro_threshold, warning_threshold):
    '''
    Parameters
    ----------
    pro1: numpy.narray, 1D
    pro2: numpy.narray, 1D
    pro3: numpy.narray, 1D
    pro_arr: numpy.narray, shape by (attention_window_size, 3)
    pro_threshold: threshold to seperate noise and debris flow for single station

    Returns:
            status: str, noise or warning
    -------

    '''

    global_mean_pro = np.sum(pro_arr[pro_arr > pro_threshold]) / pro_arr.size

    if global_mean_pro > warning_threshold:
        status = "warning"
    else:
        status = "noise"

    return status


def calculate_increased_time(date_str, input_data_year):

    df = pd.read_csv(f"{CONFIG_dir['parent_dir']}/data_input/warning_timestamp_benchmark/{input_data_year}_CD29time.txt", header=None)
    warning_time_list = np.array(df).reshape(-1)

    arr = []
    t1 = UTCDateTime(date_str)

    for step in range(warning_time_list.size):
        t2 = UTCDateTime(warning_time_list[step])
        delta_t = float(t2 - t1)
        arr.append(np.abs(delta_t))
    arr = np.array(arr)

    id = np.where(arr == np.min(np.abs(arr)))

    return np.min(np.abs(arr)), warning_time_list[id][0]


def warning1(pro_threshold, warning_threshold, attention_window_size, input_station_list,
            model_type, feature_type, input_component):

    df = merge_multi_detection(input_station_list, model_type, feature_type, input_component)

    date = np.array(df.iloc[:, 0])
    warning_status = [] # 0 noise, 1 warning
    increased_warning_time = [] # float time
    warning_ref_cd29 = [] # date string

    pro18, pro12, pro13 = np.array(df.iloc[:, 3]), np.array(df.iloc[:, 6]), np.array(df.iloc[:, 9])
    label18, label12, label13 = np.array(df.iloc[:, 1]), np.array(df.iloc[:, 4]), np.array(df.iloc[:, 7])

    for step in range(attention_window_size, len(date)):

        pro_arr = np.stack((pro18[step-attention_window_size:step],
                            pro12[step-attention_window_size:step],
                            pro13[step-attention_window_size:step]), axis=-1)

        status = warning_controller(pro_arr, pro_threshold, warning_threshold)

        if status == "warning":
            label_arr = np.stack((label18[step],
                                  label12[step],
                                  label13[step]), axis=-1)
            if np.sum(label_arr) >= 1:
                status = 1 #"real_warning"
                warning_time, ref_cd29 = calculate_increased_time(date[step])
            elif np.sum(label_arr) == 0:
                status = -1 #"false_warning"
                warning_time, ref_cd29 = "false_warning", "none"
            else:
                print(f"error {date[step]}", np.mean(np.sum(pro_arr[pro_arr > pro_threshold])))
        else:
            status = 0 #"noise"
            warning_time, ref_cd29 = "noise", "none"

        warning_status.append(status)
        increased_warning_time.append(warning_time)
        warning_ref_cd29.append(ref_cd29)

    df1 = df.iloc[attention_window_size:, :].copy()
    assert len(df1) == len(warning_status), f"check the size of df1 and warning_status"
    df1.loc[:, "warning_status"] = warning_status
    df1.loc[:, "increased_warning_time"] = increased_warning_time
    df1.loc[:, "warning_ref_cd29"] = warning_ref_cd29
    df1.to_csv(f"{CONFIG_dir['output_dir']}/dual_test_9S_warning/{model_type}_{feature_type}_{input_component}_warning_"
               f"{pro_threshold}_{warning_threshold}_{attention_window_size}.txt", sep=',', index=False, mode='w')

def warning(pro_threshold, warning_threshold, attention_window_size, input_station_list,
            model_type, feature_type, input_component, input_data_year):

    df = merge_multi_detection(input_station_list, model_type, feature_type, input_component)

    date = np.array(df.iloc[:, 0])
    warning_status = [] # 0 noise, 1 warning
    increased_warning_time = [] # float time
    warning_ref_cd29 = [] # date string

    pro18, pro12, pro13 = np.array(df.iloc[:, 2]), np.array(df.iloc[:, 4]), np.array(df.iloc[:, 6])

    for step in range(attention_window_size, len(date)):

        pro_arr = np.stack((pro18[step-attention_window_size:step],
                            pro12[step-attention_window_size:step],
                            pro13[step-attention_window_size:step]), axis=-1)

        status = warning_controller(pro_arr, pro_threshold, warning_threshold)

        if status == "warning":
            warning_time, ref_cd29 = calculate_increased_time(date[step], input_data_year)
        else:
            status = 0 #"noise"
            warning_time, ref_cd29 = "noise", "none"

        warning_status.append(status)
        increased_warning_time.append(warning_time)
        warning_ref_cd29.append(ref_cd29)

    df1 = df.iloc[attention_window_size:, :].copy()
    assert len(df1) == len(warning_status), f"check the size of df1 and warning_status"
    df1.loc[:, "warning_status"] = warning_status
    df1.loc[:, "increased_warning_time"] = increased_warning_time
    df1.loc[:, "warning_ref_cd29"] = warning_ref_cd29
    df1.to_csv(f"{CONFIG_dir['output_dir']}/dual_test_9S_warning/{model_type}_{feature_type}_{input_component}_warning_"
               f"{pro_threshold}_{warning_threshold}_{attention_window_size}.txt", sep=',', index=False, mode='w')

model_type, feature_type, input_data_year = "LSTM", "C", 2022
warning(0, 0.2, 16, ["ILL18", "ILL12", "ILL13"], model_type, feature_type, "EHZ", input_data_year)

model_type, feature_type, input_data_year = "XGBoost", "C", 2022
warning(0, 0.4, 2, ["ILL18", "ILL12", "ILL13"], model_type, feature_type, "EHZ", input_data_year)

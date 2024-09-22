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
from datetime import datetime

def write_down_warning(model_type, feature_type, input_component, warning_type, record):
    f = open(f"./output_results/{model_type}_{feature_type}_{input_component}_{warning_type}.txt", 'a')
    f.write(str(record) + "\n")
    f.close()


def merge_multi_detection(input_station_list, model_type, feature_type, input_component):
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # get the parent path

    df0 = pd.read_csv(f"{parent_dir}/output_results/predicted_results/"
                      f"{input_station_list[0]}_{model_type}_{feature_type}_{input_component}_testing_output.txt", header=0)

    df1 = pd.read_csv(f"{parent_dir}/output_results/predicted_results/"
                      f"{input_station_list[1]}_{model_type}_{feature_type}_{input_component}_testing_output.txt", header=0)

    df2 = pd.read_csv(f"{parent_dir}/output_results/predicted_results/"
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


def calculate_increased_time(date_str):
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # get the parent path

    df = pd.read_csv(f"{parent_dir}/plotting/network_warning/2020_CD29time.txt", header=None)
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


def warning(pro_threshold, warning_threshold, attention_window_size, input_station_list,
            model_type, feature_type, input_component):
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # get the parent path

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
    df1.to_csv(f"{parent_dir}/plotting/network_warning/output/{model_type}_{feature_type}_{input_component}_warning_"
               f"{pro_threshold}_{warning_threshold}_{attention_window_size}.txt", sep=',', index=False, mode='w')


def warning_summary(pro_threshold, warning_threshold, attention_window_size,
                    model_type, feature_type, input_component):
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # get the parent path
    # t1 minutess before the ILL18 start time, t2 minutes after the CD29 will not as false warning
    t1, t2 = 60, 180


    df1 = pd.read_csv(f"{parent_dir}/plotting/network_warning/output/{model_type}_{feature_type}_{input_component}_warning_"
                      f"{pro_threshold}_{warning_threshold}_{attention_window_size}.txt", header=0)
    date = np.array(df1.iloc[:, 0])

    df2 = pd.read_csv(f"{parent_dir}/plotting/network_warning/2020_testing_events.txt", header=None)
    df3 = pd.read_csv(f"{parent_dir}/plotting/network_warning/2020_CD29time.txt", header=None)

    temp = []
    false_warning = np.array(df1.iloc[:, -2])
    df1['increased_warning_time'] = df1['increased_warning_time'].replace(to_replace=['noise', 'false_warning'], value=0)

    for step in range(len(df2)):
        id1 = np.where(date == df2.iloc[step, 0])[0][0]
        id2 = np.where(date == df2.iloc[step, -1])[0][0]
        id3 = np.where(date == df3.iloc[step, 0][:-3]+":00")[0][0]

        # remove the false warning
        # 60 mins before the ILL18 start time, 180 minutes after the CD29
        false_warning[id1 - t1 : id3 + t2] = np.where(false_warning[id1 - t1 : id3 + t2] == 'false_warning',
                                                        'fake_warning', # if the donditation is Ture
                                                        false_warning[id1 - t1 : id3 + t2])# if the donditation is False

        # make sure only use the warning time before id3 (cd29 time stamps)
        # and the warning may start before the manually labeled start time
        increased_warning_time = np.array(df1.iloc[id1-20:id3+60, -2], dtype= float)
        if np.sum(increased_warning_time) > 0:
            temp.append(np.max(increased_warning_time)) # do not warry, the increased_warning_time decreases
        else:
            temp.append(0)

    false_warning_times = np.where(false_warning == 'false_warning')[0].size


    record = [model_type, feature_type, input_component,
              pro_threshold, warning_threshold, attention_window_size, "false_warning_times", false_warning_times,
              "increased_warning", np.sum(temp), np.max(temp), np.min(temp), len(temp) - np.count_nonzero(temp)]
    record.extend(list(temp))

    f = open(f"{parent_dir}/plotting/network_warning/warning_summary_{model_type}_{feature_type}_{input_component}.txt", 'a')
    f.write(str(record) + "\n")
    f.close()


    # replace the df1
    df1.iloc[:, -2] = false_warning
    df1.to_csv(f"{parent_dir}/plotting/network_warning/output/{model_type}_{feature_type}_{input_component}_warning_"
               f"{pro_threshold}_{warning_threshold}_{attention_window_size}.txt", sep=',', index=False, mode='w')

    return record


def main(model_type, feature_type, input_component):

    print(model_type, feature_type, input_component)

    pro_threshold = 0
    input_station_list = ["ILL18", "ILL12", "ILL13"]

    for idx1, warning_threshold in enumerate(np.arange(0.1, 1.1, 0.1)):
        for idx2, attention_window_size in enumerate(np.arange(1, 21, 1)):

            warning_threshold = np.round(warning_threshold, 1)
            attention_window_size = np.round(attention_window_size, 0)

            warning(pro_threshold, warning_threshold, attention_window_size,
                    input_station_list, model_type, feature_type, input_component)

            record = warning_summary(pro_threshold, warning_threshold, attention_window_size,
                                     model_type, feature_type, input_component)

            print(f"Finish, {idx1}--{idx2}, {record} {pro_threshold, warning_threshold, attention_window_size, model_type, feature_type, input_component}",
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")

    args = parser.parse_args()

    main(args.model_type, args.feature_type, args.input_component)


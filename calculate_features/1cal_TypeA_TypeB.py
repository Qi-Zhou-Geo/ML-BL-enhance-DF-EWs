#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission


import os
import argparse
from datetime import datetime

import pytz

import numpy as np
import pandas as pd

from scipy.signal import hilbert, lfilter, butter, spectrogram
from scipy.stats import kurtosis, skew, iqr

from obspy import read, Stream, read_inventory, signal
from obspy.core import UTCDateTime # default is UTC+0 time zone


from Type_A_features import *      # import Qi's all features (by *)
from Type_B_features import *      # import Clement's all features (by *)
from seismic_data_processing import * # load and process the seismic signals

def check_folder(input_year, input_station, input_component):

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    folder_path = f"{parent_dir}/data/seismic_feature/{input_year}/{input_station}/{input_component}"

    for folder_name in ["txt", "npy"]:
        if not os.path.exists(f"{folder_path}/{folder_name}"):
            os.makedirs(f"{folder_path}/{folder_name}")
        else:
            pass


def cal_attributes_B(data_array, sps): # the main function is from Clement
    # sps: sampling frequency; flag=0: one component seismic signal;
    features = calculate_all_attributes(Data=data_array, sps=sps, flag=0)[0] # feature 1 to 60
    feature_array = features[1:]# leave features[0]=event_duration
    return feature_array # 59 features

def cal_attributes_A(data_array, ruler=300): # the main function is from Qi
    data_array_nm = data_array * 1e9 # converty m/s to nm/s
    feature_array = calBL_feature(data_array_nm, ruler)

    return feature_array # 17 features


def record_data_header(input_year, input_station, input_component, julday):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    folder_path = f"{parent_dir}/data/seismic_feature/{input_year}/{input_station}/{input_component}/txt/"

    featureName1 = ['time_window_start', 'time_stamps', 'station', 'component',
                    'RappMaxMean', 'RappMaxMedian', 'AsDec', 'KurtoSig','KurtoEnv', 'SkewnessSig','SkewnessEnv',
                    'CorPeakNumber', 'INT1', 'INT2', 'INT_RATIO', 'ES_0', 'ES_1', 'ES_2', 'ES_3', 'ES_4', 'KurtoF_0',
                    'KurtoF_1', 'KurtoF_2', 'KurtoF_3', 'KurtoF_4', 'DistDecAmpEnv','env_max_to_duration', 'RMS', 'IQR', 'MeanFFT', 'MaxFFT',
                    'FmaxFFT', 'FCentroid', 'Fquart1', 'Fquart3', 'MedianFFT', 'VarFFT', 'NpeakFFT', 'MeanPeaksFFT', 'E1FFT','E2FFT','E3FFT', 'E4FFT',
                    'gamma1', 'gamma2', 'gammas', 'SpecKurtoMaxEnv', 'SpecKurtoMedianEnv', 'RatioEnvSpecMaxMean', 'RatioEnvSpecMaxMedian','DistMaxMean',
                    'DistMaxMedian', 'NbrPeakMax', 'NbrPeakMean', 'NbrPeakMedian','RatioNbrPeakMaxMean', 'RatioNbrPeakMaxMedian',
                    'NbrPeakFreqCenter', 'NbrPeakFreqMax', 'RatioNbrFreqPeaks','DistQ2Q1', 'DistQ3Q2', 'DistQ3Q1'] # RF features

    featureName2 = ['time_window_start', 'time_stamps', 'station', 'component',
                    'digit1','digit2','digit3','digit4','digit5', 'digit6','digit7','digit8','digit9',
                    'max', 'min', 'iqr', 'goodness', 'alpha', 'ks', 'MannWhitneU', 'follow'] # BL features

    featureName1, featureName2 = np.array(featureName1), np.array(featureName2)

    # give features title and be careful the file name and path
    with open(f"{folder_path}{input_year}_{input_station}_{input_component}_{julday}_B.txt", 'a') as file1:
        np.savetxt(file1, [featureName1], header='', delimiter=',', comments='', fmt='%s')
    # give features title and be careful the file name and path
    with open(f"{folder_path}{input_year}_{input_station}_{input_component}_{julday}_A.txt", 'a') as file2:
        np.savetxt(file2, [featureName2], header='', delimiter=',', comments='', fmt='%s')


def record_data(input_year, input_station, input_component, arr, feature_type, julday):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    folder_path = f"{parent_dir}/data/seismic_feature/{input_year}/{input_station}/{input_component}/txt/"

    arr = arr.reshape(1, -1)
    # seismic features to be saved
    if feature_type == "RF":
        with open(f"{folder_path}{input_year}_{input_station}_{input_component}_{julday}_B.txt", 'a') as file:
            np.savetxt(file, arr, header='', delimiter=',', comments='', fmt='%s')#'%.4f')
    elif feature_type == "BL":
        with open(f"{folder_path}{input_year}_{input_station}_{input_component}_{julday}_A.txt", 'a') as file:
            np.savetxt(file, arr, header='', delimiter=',', comments='', fmt='%s')#'%.4f')
    else:
        print("error at record_data")


def loop_time_step(st, input_year, input_station, input_component, input_window_size, julday):

    # columns array to store the seismic data for network features
    sps = int(st[0].stats.sampling_rate)
    seismic_array = np.empty((0, sps * input_window_size))  # creat the 4identifiers + data in (input_window_size * SPS + 1)
    d = UTCDateTime(year=input_year, julday=julday)  # the start day, e.g.2014-07-12 00:00:00

    for step in range(0, int(3600 * 24 / input_window_size)):  # 1440 = 24h * 60min
        d1 = d + (step) * input_window_size  # from minute 0, 1, 2, 3, 4
        d2 = d + (step + 1) * input_window_size  # from minute 1, 2, 3, 4, 5

        # float timestamps, mapping station and component
        time = datetime.fromtimestamp(d1.timestamp, tz=pytz.utc)
        time = time.strftime('%Y-%m-%d %H:%M:%S')
        id = np.array([time, d1.timestamp, input_station, input_component])

        tr = st.copy()
        try:  # hava data in this time domain d1 to d2
            tr.trim(starttime=d1, endtime=d2, nearest_sample=False)
            seismic_data = tr[0].data[:sps * input_window_size]
            type_B_arr = cal_attributes_B(data_array=seismic_data, sps=sps)
            type_A_arr = cal_attributes_A(data_array=seismic_data)
        except Exception as e:  # NOT hava any data in this time domain d1 to d2
            seismic_data = np.full(input_window_size * sps, np.nan)
            type_B_arr = np.full(59, np.nan)
            type_A_arr = np.full(17, np.nan)
            print(f"NaN in {input_year}-{input_station}-{input_component}-{time}, \n"
                  f"Exception {e} \n")


        arr_RF, arr_BL = np.append(id, type_B_arr), np.append(id, type_A_arr)
        record_data(input_year, input_station, input_component, arr_RF, "RF", julday)  # write data by custom function
        record_data(input_year, input_station, input_component, arr_BL, "BL", julday)  # write data by custom function

        seismic_array = np.vstack((seismic_array, seismic_data))

    # save the npy every julday
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    folder_path = f"{parent_dir}/data/seismic_feature/{input_year}/{input_station}/{input_component}/npy/"
    np.save(f"{folder_path}/{input_year}_{input_station}_{input_component}_{julday}.npy", seismic_array)


def cal_loop(seismic_network, input_year, input_station, input_component, input_window_size, id1, id2):

    for julday in range(id1, id2):

        julday = str(julday).zfill(3)
        data_start = UTCDateTime(year=input_year, julday=julday)
        data_end = data_start + 24 * 3600

        st = load_seismic_signal(seismic_network, input_station, input_component, data_start, data_end)

        # write the seismic features header
        record_data_header(input_year, input_station, input_component, julday)

        loop_time_step(st, input_year, input_station, input_component, input_window_size, julday)


def main(seismic_network, input_year, input_station, input_component, input_window_size, id):
    '''
    Set the input parametres for main function

    Parameters:
    - input_year (int or string): calculate year
    - input_station (str): calculate seismic station
    - input_component (str): component of calculate seismic station
    - input_window_size (int): window size of calculate

    Returns:
    - No returns
    '''

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"Start Job {job_id}: {input_year}, {input_station}: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S") )

    # check the folder
    try:
        check_folder(input_year, input_station, input_component)
    except FileExistsError as e:
        print(f"{seismic_network}, {input_year}, {input_station}, {input_component}, {input_window_size}, {id}, \n"
              f"Exception {e}: Directory already exists, ignoring.")
    except Exception as e:
        print(f"{seismic_network, input_year, input_station, input_component, input_window_size, id}, \n"
              f"Exception {e}")

    #map_start_julday = {2013:147, 2014:91,  2017:140, 2018:145, 2019:145, 2020:152}
    #map_end_julday   = {2013:245, 2014:273, 2017:183, 2018:250, 2019:250, 2020:250}
    #id1, id2 = map_start_julday.get(input_year),  map_end_julday.get(input_year)

    id1, id2 = id, id + 1
    cal_loop(seismic_network, input_year, input_station, input_component, input_window_size, id1, id2)


    print(f"End Job {job_id}: {input_year}, {input_station}: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seismic_network", type=str, default=2020, help="check the year")
    parser.add_argument("--input_year", type=int, default=2020, help="check the year")
    parser.add_argument("--input_station", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_component", type=str, default="EHZ", help="check the input_component")
    parser.add_argument("--input_window_size", type=int, default=60, help="check the calculate window size")
    parser.add_argument("--id", type=int, default=60, help="check the calculate window size")

    args = parser.parse_args()
    main(args.seismic_network, args.input_year, args.input_station, args.input_component, args.input_window_size, args.id)

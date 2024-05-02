#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
# re-written by Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# reference to (1) Chmiel, Małgorzata et al. (2021): e2020GL090874, (2) Floriane Provost et al.(2017): 113-120.


# <editor-fold desc="**** load the package">
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

import sys
import platform

if platform.system() == "Darwin": # your local PC name
    sys.path.append('/Users/qizhou/#python/functions/')
elif platform.system() == "Linux":  # your remote server name
    sys.path.append('/storage/vast-gfz-hpc-01/home/qizhou/2python/functions/')

from Type_A_features import *      # import Qi's all features (by *) codes/functions
from Type_B_features import *      # import Clement's all features (by *) codes/functions



# <editor-fold desc="**1** set input-output path">
def set_in_out_path (input_year, input_station, input_component, input_window_size):
    '''
    Set the input and output path

    Parameters:
    - input_year (int or str): calculate year
    - input_station (str): calculate seismic station
    - input_component (str): component of calculate seismic station
    - input_window_size (int): window size of calculate

    Returns:
    - OUTPUT_DIR (str): global parameters, output path for seismic features
    - SAC_DIR (str): global parameters, input path for seismic data
    '''

    OUTPUT_DIR, SAC_DIR = "check-the-path", "check-the-path"
    if platform.system() == "Darwin":  # your local Mac PC name
        SAC_DIR, OUTPUT_DIR = "/Users/qizhou/Downloads/", "/Users/qizhou/Downloads/"
    elif platform.system() == "Linux":  # your remote server name
        SAC_DIR = f"/home/qizhou/0data/{input_year}/{input_station}/{input_component}/"
        OUTPUT_DIR = f"/home/qizhou/1projects/dataForML/out{input_window_size}/{input_year}/"

    return OUTPUT_DIR, SAC_DIR
# </editor-fold>


# <editor-fold desc="**2** read seismic data">
def load_seismic_signal(data_name):
    '''
    Load the seismic signal

    Parameters:
    - data_name (str): seismic data name, exclude the seismic path

    Returns:
    - st (stream): seismic stream
    - stEnvlope (stream): seismic stream that the amplitude convert to envelop
    '''
    
    st = read(SAC_DIR + data_name)
    st.merge(method=1, fill_value=0)
    st.detrend('linear')
    st.detrend('demean')
    st.filter("bandpass", freqmin=1, freqmax=45)
    st._cleanup()

    # envolope, usually the left/right side is not reliable
    stEnvlope = st.copy()
    hilb = hilbert(stEnvlope[0].data)
    stEnvlope[0].data= (stEnvlope[0].data ** 2 + hilb ** 2) ** 0.5
    # replace the first and last 3 minutes data by abs
    stEnvlope[0].data[:3 * SPS] = np.abs(st[0].data[:3 * SPS])
    stEnvlope[0].data[-3 * SPS:] = np.abs(st[0].data[-3 * SPS:])
    
    return st, stEnvlope
# </editor-fold>


# <editor-fold desc="**3** calculate features (based on Clement's + Qi's codes)">
def cal_attributesRF(data_array): # the main function is from Clement
    # sps: sampling frequency; flag=0: 1 component seismic signal;
    features = calculate_all_attributes(Data=data_array, sps=SPS, flag=0)[0] # feature 1 to 60
    feature_array = features[1:]# leave features[0]=event_duration
    return feature_array # 59 features

def cal_attributesBL(data_array, RULER): # the main function is from Qi
    feature_array = calBL_feature(data_array, RULER)
    return feature_array # 17 features


def record_data_header(input_year, input_station, input_component):

    featureName1 = ['timeDate', 'time_stamps', 'station', 'component',
                      'RappMaxMean', 'RappMaxMedian', 'AsDec', 'KurtoSig','KurtoEnv', 'SkewnessSig','SkewnessEnv',
                      'CorPeakNumber', 'INT1', 'INT2', 'INT_RATIO', 'ES_0', 'ES_1', 'ES_2', 'ES_3', 'ES_4', 'KurtoF_0',
                      'KurtoF_1', 'KurtoF_2', 'KurtoF_3', 'KurtoF_4', 'DistDecAmpEnv','env_max_to_duration', 'RMS', 'IQR', 'MeanFFT', 'MaxFFT',
                      'FmaxFFT', 'FCentroid', 'Fquart1', 'Fquart3', 'MedianFFT', 'VarFFT', 'NpeakFFT', 'MeanPeaksFFT', 'E1FFT','E2FFT','E3FFT', 'E4FFT',
                      'gamma1', 'gamma2', 'gammas', 'SpecKurtoMaxEnv', 'SpecKurtoMedianEnv', 'RatioEnvSpecMaxMean', 'RatioEnvSpecMaxMedian','DistMaxMean',
                      'DistMaxMedian', 'NbrPeakMax', 'NbrPeakMean', 'NbrPeakMedian','RatioNbrPeakMaxMean', 'RatioNbrPeakMaxMedian',
                      'NbrPeakFreqCenter', 'NbrPeakFreqMax', 'RatioNbrFreqPeaks','DistQ2Q1', 'DistQ3Q2', 'DistQ3Q1'] # RF features
    featureName2 = ['timeDate', 'time_stamps', 'station', 'component',
                      'digit1','digit2','digit3','digit4','digit5',
                     'digit6','digit7','digit8','digit9',
                     'max', 'min', 'iq', 'goodness', 'alpha', 'ks', 'MannWhitneU', 'followOrNot','FD'] # BL features

    featureName1, featureName2 = np.array(featureName1), np.array(featureName2)

    # give features title and be careful the file name and path
    with open(f"{OUTPUT_DIR}{input_year}_{input_station}_{input_component}_RF.txt", 'a') as file1:
        np.savetxt(file1, [featureName1], header='', delimiter=',', comments='', fmt='%s')
    # give features title and be careful the file name and path
    with open(f"{OUTPUT_DIR}{input_year}_{input_station}_{input_component}_BL.txt", 'a') as file2:
        np.savetxt(file2, [featureName2], header='', delimiter=',', comments='', fmt='%s')


def record_data(input_year, input_station, input_component, arr, RF_or_BL):
    arr = arr.reshape(1, -1)
    # seismic features to be saved
    if RF_or_BL == "RF":
        with open(f"{OUTPUT_DIR}{input_year}_{input_station}_{input_component}_RF.txt", 'a') as file:
            np.savetxt(file, arr, header='', delimiter=',', comments='', fmt='%s')#'%.4f')
    elif RF_or_BL == "BL":
        with open(f"{OUTPUT_DIR}{input_year}_{input_station}_{input_component}_BL.txt", 'a') as file:
            np.savetxt(file, arr, header='', delimiter=',', comments='', fmt='%s')#'%.4f')
    else:
        print("error at record_data")
# </editor-fold>


# <editor-fold desc="**4** run the calculator">
def cal_loop (input_year, input_station, input_component, input_window_size):#(input_year, cal_window, station, component, id1, id2):

    # Mapping for ILL stations and components, default to -1 if not found station or component
    stationMapping = {'ILL18': 8.0, 'ILL12': 2.0, 'ILL13': 3.0, 'IGB01': 8.0, 'IGB02': 2.0, 'IGB10':3.0}
    componentMapping = {'EHZ': 1.0, 'EHE': 2.0, 'EHN': 3.0, 'BHZ': 1.0, 'BHE': 2.0, 'BHN': 3.0}

    # list all available sac file
    all_sacFile = os.listdir(SAC_DIR)

    # write the seismic features header
    record_data_header(input_year, input_station, input_component)

    for julday in range(id1, id2):  # 91 = 1st of May to 305=31 of Nov.
        julday = str(julday).zfill(3)
        miniSeed_name = f"{miniSEED}.{julday}"

        # columns array to store the seismic data for network features
        seismic_array = np.empty((0, SPS*input_window_size+3))# creat the 3identifiers + data in input_window_size*SPS

        d = UTCDateTime(year=input_year, julday=julday)  # the start day, e.g.2014-07-12 00:00:00

        if miniSeed_name in all_sacFile:  # Yes, have seismic data in this julian day
            st, stEnvlope = load_seismic_signal(data_name=miniSeed_name)

            for step in range(0, int(60*60*24/input_window_size)):  # 1440 = 24h * 60min
                d1 = d + (step) * input_window_size     # from minute 0, 1, 2, 3, 4
                d2 = d + (step + 1) * input_window_size # from minute 1, 2, 3, 4, 5

                # float timestamps, mapping station and component
                timeDate = datetime.fromtimestamp(d2.timestamp, tz=pytz.utc)
                timeDate = timeDate.strftime('%Y-%m-%d %H:%M:%S')
                id = np.array([timeDate, d2.timestamp, componentMapping.get(input_component, -1), stationMapping.get(input_station, -1)])

                tr1 = st.copy()
                tr2 = stEnvlope.copy()
                try:  # hava data in this time domain d1 to d2
                    tr1.trim(starttime=d1, endtime=d2, nearest_sample=False)
                    tr2.trim(starttime=d1, endtime=d2, nearest_sample=False)
                    # calculate the RF features
                    if len(tr1[0].data) >= SPS*input_window_size : # full time window with 100data * 120s
                        seismic = tr1[0].data[:input_window_size * SPS]
                        arr1 = cal_attributesRF(data_array=seismic)
                    else: # incomplete data ！= 100data * 120s
                        seismic, arr1 = np.full(input_window_size * SPS, np.nan), np.full(59, np.nan)
                    # calculate the BL features
                    benford = cal_attributesBL(data=tr2[0].data, ruler=RULER)
                    # calculate the FD features
                    fd = calFD_feature(data=tr1[0].data, SPS=SPS)
                    BF_FD = np.append(benford, fd)
                except:  # NOT hava any data in this time domain d1 to d2
                    # calculate the RF features
                    seismic, arr1 = np.full(input_window_size * SPS, np.nan), np.full(59, np.nan)
                    # calculate the BL features
                    benford = np.full(17, np.nan)
                    fd = np.full(1, np.nan)
                    BF_FD = np.append(benford, fd)

                arr_RF, arr_BL_FD = np.append(id, arr1), np.append(id, BF_FD)
                record_data(input_year, input_station, input_component, arr_RF,    "RF") # write data by custom function
                record_data(input_year, input_station, input_component, arr_BL_FD, "BL") # write data by custom function

                waveforms = np.append(id[1:].astype(float), seismic)
                seismic_array = np.vstack((seismic_array, waveforms))

        else:  # DoNOT, have any seismic data in this julian day
            for step in range(0, int(60*60*24/input_window_size)):  # 1440 = 24h * 60min
                #d1 = d + (step) * input_window_size     # from min 0, 1, 2, 3, 4
                d2 = d + (step + 1) * input_window_size  # from min 1, 2, 3, 4, 5

                # float timestamps, mapping station and component
                timeDate = datetime.fromtimestamp(d2.timestamp, tz=pytz.utc)
                timeDate = timeDate.strftime('%Y-%m-%d %H:%M:%S')
                id = np.array([timeDate, d2.timestamp, componentMapping.get(input_component, -1), stationMapping.get(input_station, -1)])

                # calculate the RF features
                seismic, arr1 = np.full(input_window_size * SPS, np.nan), np.full(59, np.nan)
                # calculate the BL features
                benford = np.full(17, np.nan)
                fd = np.full(1, np.nan)
                BF_FD = np.append(benford, fd)

                arr_RF, arr_BL_FD = np.append(id, arr1), np.append(id, BF_FD)
                record_data(input_year, input_station, input_component, arr_RF,    "RF") # write data by custom function
                record_data(input_year, input_station, input_component, arr_BL_FD, "BL")  # write data by custom function

                waveforms = np.append(id[1:].astype(float), seismic)
                seismic_array = np.vstack((seismic_array, waveforms))

        print(f"done: {miniSeed_name}")

        # save the npy every julday
        np.save(f"{OUTPUT_DIR}/{input_station}/{input_component}/{input_year}_{input_station}_{input_component}_{julday}.npy", seismic_array)

# </editor-fold>


def main(input_year, input_station, input_component, input_window_size):
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

    global SPS, OUTPUT_DIR, SAC_DIR, RULER, miniSEED

    yearMappingStart = {2013:147, 2014:91, 2017:137, 2018:135, 2019:135, 2020:149}
    yearMappingEnd = {2013:245, 2014:273, 2017:215, 2018:255, 2019:291, 2020:275}


    if year in [2013, 2014]:  # 2013-2014 data
        OUTPUT_DIR, SAC_DIR = set_in_out_path(input_year, input_station, input_component, input_window_size)
        SPS = 200
        RULER = 1e3
        miniSEED = f"GM.{input_station}.{input_component}.{input_year}"
    elif year not in [2013, 2014]:  # 2017-2020 data
        OUTPUT_DIR, SAC_DIR = set_in_out_path(year, input_station, input_component, input_window_size)
        SPS = 100
        RULER = 1e2
        miniSEED = f"9S.{input_station}.{input_component}.{input_year}"
    else:
        print("error in OUTPUT_DIR, SAC_DIR = set_in_out_path")


    id1, id2 = yearMappingStart.get(input_year, 91),  yearMappingEnd.get(input_year, 300)
    cal_loop(input_year, input_window_size, input_station, input_component, id1=id1, id2=id2)

    print(f"End Job {job_id}: {input_year}, {input_station}: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_year", type=int, default=2020, help="check the year")
    parser.add_argument("--input_station", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_component", type=str, default="EHZ", help="check the input_component")
    parser.add_argument("--input_window_size", type=int, default=60, help="check the calculate window size")

    args = parser.parse_args()
    main(args.input_year, args.input_station, args.input_component, args.input_window_size)

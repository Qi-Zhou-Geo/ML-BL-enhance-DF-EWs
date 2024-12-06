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
from itertools import combinations

import scipy
from scipy.signal import hilbert, lfilter, butter, spectrogram, coherence, correlate, correlation_lags
from scipy.stats import kurtosis, skew, iqr, wasserstein_distance

from obspy import read, Stream, read_inventory, signal
from obspy.core import UTCDateTime # default is UTC+0 time zone


# <editor-fold desc="define the parent directory">
import platform
if platform.system() == 'Darwin':
    parent_dir = "/Users/qizhou/#python/#GitHub_saved/ML-BL-enhance-DF-EWs"
elif platform.system() == 'Linux':
    parent_dir = "/home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs"
else:
    print(f"check the parent_dir for platform.system() == {platform.system()}")
# add the parent_dir to the sys
import sys
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
else:
    pass
# </editor-fold>


# import the custom functions
from config.config_dir import CONFIG_dir, path_mapping


def check_folder(seismic_network, input_year, input_station, input_component):

    folder_path = f"{CONFIG_dir['feature_output_dir']}/{path_mapping(seismic_network)}/{input_year}/{input_component}_net"
    CONFIG_dir['net_txt_dir'] = folder_path
    os.makedirs(f"{CONFIG_dir['net_txt_dir']}", exist_ok=True)


def rms_network(rms1, rms2, rms3, iqr1, iqr2, iqr3):
    # rms1, rms2, rms3 is from ILL18(8.0), ILL12(2.0), ILL13(3.0)
    # iqr1, iqr2, iqr3 is from ILL18(8.0), ILL12(2.0), ILL13(3.0)
    # station_mapping = {'ILL18': 8.0, 'ILL12': 2.0, 'ILL13': 3.0, 'IGB01': 8.0, 'IGB02': 2.0, 'IGB10': 3.0}
    station_mapping_reverse = {0:8.0, 1:2.0, 2:3.0}

    max_rms = np.max([rms1, rms2, rms3])
    id_max = np.argmax([rms1, rms2, rms3])# find max id
    id_max1 = station_mapping_reverse.get(id_max, -1)

    min_rms = np.min([rms1, rms2, rms3])
    id_min = np.argmin([rms1, rms2, rms3])# find min id
    id_min1 = station_mapping_reverse.get(id_min, -1)

    ratio_rms = max_rms / min_rms

    min_iqr = np.min([iqr1, iqr2, iqr3])
    if min_iqr != 0 :
        ratio_iqr = np.max([iqr1, iqr2, iqr3]) / min_iqr
    else:
        ratio_iqr = 1

    return id_max1, id_min1, ratio_rms, ratio_iqr

def wd_lagTime_coher(trace1, trace2):
    # the sampling frequency of trace is 0.01 (100 data 1 second)
    if cal_year != 2013 or cal_year != 2014:
        sampling_frequency = 0.01
    else :
        sampling_frequency = 0.005
    # calculate the wasserstein distance
    normalized_trace1 = trace1 ** 2 / np.sum(trace1 ** 2)
    normalized_trace2 = trace2 ** 2 / np.sum(trace2 ** 2)
    wd = wasserstein_distance(normalized_trace1, normalized_trace2)

    # calculate the lag time of trace1 and trace2
    correlation = correlate(trace1, trace2, method='fft', mode="full")# cross-correlate trace1 and trace2
    lags = correlation_lags(trace1.size, trace2.size, mode="full")
    # lag_time -> positive means trace1 is later trace2, negative is trace1 earlier than trace2
    lagTime = lags[np.argmax(correlation)] * sampling_frequency # unit is 0.01 second (sampling frequency)

    # Calculate cross-coherence by Welch’s method
    frequency, coherence_values = coherence(trace1, trace2, fs=1)

    return wd, lagTime, coherence_values

def Coher(trace1, trace2):  # cross-coherence by Gosia

    f = trace1 ** 2 / np.sum(trace1 ** 2)
    g = trace2 ** 2 / np.sum(trace2 ** 2)
    wd = wasserstein_distance(f, g)

    lentrace = len(trace1)
    maxlag = lentrace
    goodnumber = int(2 ** (np.ceil(np.log2(lentrace)) + 2))

    tr2 = np.zeros(goodnumber)
    tr2[0:lentrace] = trace2
    tr2 = scipy.fftpack.fft(tr2, overwrite_x=True)
    tr2.imag *= -1
    tr1 = np.zeros(goodnumber)
    tr1[maxlag:maxlag + lentrace] = trace1
    tr1 = scipy.fftpack.fft(tr1, overwrite_x=True)

    try:
        tr_cc = (tr1 * tr2) / (np.absolute(tr1) * np.absolute(tr2))
    except:# Calculate the tr_cc array, handling invalid values
        tr_cc = np.divide(tr1 * tr2, np.absolute(tr1) * np.absolute(tr2), out=np.zeros_like(tr1), where=(tr1 != 0) & (tr2 != 0) )
        print("method 2")

    tr_cc[np.isnan(tr2)] = 0.0
    tr_cc[np.isinf(tr2)] = 0.0

    go = scipy.fftpack.ifft(tr_cc, overwrite_x=True)[0:2 * maxlag + 1].real

    coherence_values = np.max(go)
    lagTime = np.argmax(go)
    return wd, lagTime, coherence_values


def record_data_header(input_year, input_component, julday):

    feature_names = ['time_window_start', 'time_stamps', 'component',
                     'id_maxRMS', 'id_minRMS',
                     'ration_maxTOminRMS', 'ration_maxTOminIQR',
                     'mean_coherenceOfNet', 'max_coherenceOfNet',
                     'mean_lagTimeOfNet', 'std_lagTimeOfNet',
                     'mean_wdOfNet', 'std_wdOfNet']  # timestamps + component + 9 network features

    feature_names = np.array(feature_names)


    # give features title and be careful the file name and path
    with open(f"{CONFIG_dir['net_txt_dir']}/{input_year}_{input_component}_{julday}_net.txt", 'w') as file:
        np.savetxt(file, [feature_names], header='', delimiter=',', comments='', fmt='%s')


def run_cal_loop(seismic_network, input_year, input_component, input_window_size, id1, id2, station_list):

    for julday in range(id1, id2):  # 91 = 1st of May to 305=31 of Nov.
        d = UTCDateTime(year=input_year, julday=julday)  # the start day, e.g.2014-07-12 00:00:00
        julday = str(julday).zfill(3)

        # write the seismic features header
        record_data_header(input_year, input_component, julday)

        # load data then input to cal_loop
        np_dir = f"{CONFIG_dir['feature_output_dir']}/{path_mapping(seismic_network)}/" \
                 f"{input_year}/{station_list[0]}/{input_component}/npy" # set np_dir
        dataILL18 = np.load(f"{np_dir}/{input_year}_{station_list[0]}_{input_component}_{julday}.npy")  # 3identifiers + rms + 12001data

        np_dir = f"{CONFIG_dir['feature_output_dir']}/{path_mapping(seismic_network)}/" \
                 f"{input_year}/{station_list[1]}/{input_component}/npy"  # set np_dir
        dataILL12 = np.load(f"{np_dir}/{input_year}_{station_list[1]}_{input_component}_{julday}.npy")  # 3identifiers + rms + 12001data

        np_dir = f"{CONFIG_dir['feature_output_dir']}/{path_mapping(seismic_network)}/" \
                 f"{input_year}/{station_list[2]}/{input_component}/npy"  # set np_dir
        dataILL13 = np.load(f"{np_dir}/{input_year}_{station_list[2]}_{input_component}_{julday}.npy")  # 3identifiers + rms + 12001data

        ####################################

        # check the data length
        if len(dataILL18) != len(dataILL12) or len(dataILL18) != len(dataILL13) or len(dataILL12) != len(dataILL13):
            print("check the input data length: ", input_year, julday)
            print(len(dataILL18))
            print(len(dataILL12))
            print(len(dataILL13))
            exit()
        else:
            arra_length = len(dataILL18)

        for step in range(arra_length): # if len(dataILL18) != len(dataILL12) make sure all data are same length

            # processed 1-min seismic data array
            waveformILL18, waveformILL12, waveformILL13 = dataILL18[step, :], dataILL12[step, :], dataILL13[step, :]
            rms18, rms12, rms13 = np.sqrt(np.mean(waveformILL18 ** 2)), np.sqrt(np.mean(waveformILL12 ** 2)), np.sqrt(np.mean(waveformILL13 ** 2))
            iqr18, iqr12, iqr13 = iqr(np.abs(waveformILL18)), iqr(np.abs(waveformILL12)), iqr(np.abs(waveformILL13))
            # first three RMS related network features
            id_max, id_min, ratio_rms, ratio_iqr = rms_network(rms18, rms12, rms13, iqr18, iqr12, iqr13)


            pair_combinations = combinations((waveformILL18, waveformILL12, waveformILL13),2)  # from [a,b,c] to (a,b), (a,c), (b,c)

            wd_list, lagTime_list, coherence_list = [], [], []
            for pair in pair_combinations:
                xcorr = Coher(trace1=pair[0], trace2=pair[1])  # Gosia's code

                wd_list.append(xcorr[0])
                lagTime_list.append(xcorr[1])
                coherence_list.append(np.max(xcorr[2]))  # only select the max coherence values between two stations

            time_stamps = float(d + (step) * input_window_size)
            time = datetime.fromtimestamp(time_stamps, tz=pytz.utc)
            time = time.strftime('%Y-%m-%dT%H:%M:%S')

            mean_coherence = np.mean(coherence_list)  # coherence related network features
            max_coherence = np.max(coherence_list)

            mean_lagTime = np.mean(lagTime_list)  # lagTime related network features
            std_lagTime = np.std(lagTime_list)

            mean_wd = np.mean(wd_list)  # wd related network features
            std_wd = np.std(wd_list)

            arr = np.array((time, time_stamps, input_component,
                            id_max, id_min,
                            ratio_rms, ratio_iqr,
                            mean_coherence, max_coherence,
                            mean_lagTime, std_lagTime,
                            mean_wd, std_wd))
            # do not give header here ( see [feature_names] )
            with open(f"{CONFIG_dir['net_txt_dir']}/{input_year}_{input_component}_{julday}_net.txt", 'a') as file:
                np.savetxt(file, arr.reshape(1, -1), header='', delimiter=',', comments='', fmt='%s')#fmt_list)


def main(seismic_network, input_year, station_list, input_component, input_window_size, id):  # Update the global variables with the values from command-line arguments
    print(f"Start Job: {input_year}, {input_component}: ", datetime.now().strftime("%Y-%m-%dT%H:%M:%S") )

    # check the folder
    try:
        check_folder(seismic_network, input_year, None, input_component)
    except FileExistsError as e:
        print(f"{input_year}, {input_component}, {input_window_size}, {id}, \n"
              f"Exception {e}: Directory already exists, ignoring.")
    except Exception as e:
        print(f"{input_year}, {input_component}, {input_window_size}, {id}, \n"
              f"Exception {e}")

    #map_start_julday = {2013:147, 2014:91,  2017_1:140, 2018-2019:145, 2019:145, 2020:152}
    #map_end_julday   = {2013:245, 2014:273, 2017_1:183, 2018-2019:250, 2019:250, 2020:250}
    #id1, id2 = map_start_julday.get(input_year),  map_end_julday.get(input_year)
    if len(station_list) == 2:
        station_list.append(station_list[0])
    else:
        pass

    id1, id2 = id, id + 1
    run_cal_loop(seismic_network, input_year, input_component, input_window_size, id1, id2, station_list)


    print(f"End Job: {input_year}, {input_component}: ", datetime.now().strftime("%Y-%m-%dT%H:%M:%S") )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seismic_network", type=str, default="9S", help="check the year")
    parser.add_argument("--input_year", type=int, default=2020, help="cal_year")
    parser.add_argument("--station_list", nargs='+', type=str, help="list of stations")
    parser.add_argument("--input_component", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_window_size", type=int, default=60, help="check the calculate window size")
    parser.add_argument("--id", type=int, default=60, help="check the calculate window size")

    args = parser.parse_args()
    main(args.seismic_network, args.input_year, args.station_list, args.input_component, args.input_window_size, args.id)

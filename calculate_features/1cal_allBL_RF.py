#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
# re-written by Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# reference to (1) Chmiel, Małgorzata et al. (2021): e2020GL090874, (2) Floriane Provost et al.(2017): 113-120.
# <editor-fold desc="**** load the package">
import os
import argparse
from datetime import datetime

import pytz
import platform

import numpy as np
import pandas as pd

from scipy.signal import hilbert, lfilter, butter, spectrogram
from scipy.stats import kurtosis, skew, iqr

from obspy import read, Stream, read_inventory, signal
from obspy.core import UTCDateTime # default is UTC+0 time zone
import matplotlib.pyplot as plt

# make sure the clement.py is in the same folder of this python script
import sys
sys.path.insert(1, "/home/qizhou/1projects/dataForML/manuallyEvent/")
from ClementFor_1cal_allBL_RF import * # import Clement's all features (by *) codes/functions
#from QiFor_1cal_allBL_RF import *      # import Qi's all features (by *) codes/functions


#BL functions
import numpy as np
import pandas as pd
from scipy.stats import iqr, ks_2samp, chi2_contingency, mannwhitneyu


# the BL is based on the following refrence
# Sambridge, Malcolm, Hrvoje Tkalčić, and A. Jackson. "Benford's law in the natural sciences." Geophysical research letters 37.22 (2010).
# Zhou, Qi, et al. "Benford's law as mass movement detector in seismic signals." (2023).
def calBL_feature(data, ruler=1e1):
    '''
    Parameters
    ----------
    data: np array data after dtrend+dmean+filter 1-45 Hz, or raw data
    ruler: discard the "small" data if it less than "ruler"
    Returns: np.array
    -------
    '''
    data = np.abs(data)
    #hilb = hilbert(data)  # envolope
    #data = (data ** 2 + hilb ** 2) ** 0.5
    # BL theoretical value
    BL_frequency = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    dataSelected = data[data != 0] + ruler

    # <editor-fold desc="iq, max, min">
    iq = float("{:.2f}".format(iqr(dataSelected)))
    max_amp = float("{:.2f}".format(np.max(dataSelected)))
    min_amp = float("{:.2f}".format(np.min(dataSelected)))
    # </editor-fold>

    # <editor-fold desc="digit frequency">
    amp_data = pd.DataFrame(dataSelected)
    amp_data = amp_data.astype(str)

    d = (amp_data.iloc[:, 0]).str[0: 1]
    d = list(d)

    digit_count = np.empty((0, 9))
    for digit in range(1, 10):
        first_digit = d.count(str(digit))
        digit_count = np.append(digit_count, first_digit)

    digit_frequency = digit_count / np.sum(digit_count)
    digit_frequency = [float('{:.3f}'.format(i)) for i in digit_frequency]
    # </editor-fold>

    # <editor-fold desc="get goodness, ks, chi-squared, alpha">
    frequency = np.empty((0, 9))
    for a in range(0, 9):
        first_digit_frequency = pow((digit_frequency[a] - BL_frequency[a]), 2) / BL_frequency[a]
        frequency = np.append(frequency, first_digit_frequency)
    goodness = (1 - pow(sum(frequency), 0.5)) * 100
    goodness = float("{:.3f}".format(goodness))

    statistic, pvalue = ks_2samp(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
    ks = float("{:.3f}".format(pvalue))  # pvalue

    # chi squared methods not works well
    # digit_frequency = [x / sum(digit_frequency) for x in digit_frequency] # for chi squared
    #statistic, pvalue = chisquare(f_obs=digit_frequency, f_exp=BL_frequency)
    #chi = float("{:.3f}".format(pvalue)) # pvalue

    statistic, pvalue = mannwhitneyu(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
    MannWhitneU = float("{:.3f}".format(pvalue))  # pvalue

    if ks >= 0.95 and MannWhitneU >= 0.95:
        follow = 1 # follow BL
    else:
        follow = 0 # do not follow BL

    sum_d = []
    y_min = np.nanmin(dataSelected)
    if y_min == 0: # in csse of "divide by zero encountered in scalar divide"
        y_min = 1
    else:
        pass

    for s in range(0, len(dataSelected)):
        i = np.log(dataSelected[s] / y_min)
        sum_d.append(i)
    alpha = 1 + len(dataSelected) / np.sum(sum_d)
    alpha = float("{:.4f}".format(alpha))
    # </editor-fold>

    output = np.array([max_amp, min_amp, iq, goodness, alpha, ks, MannWhitneU, follow], dtype=float)
    output = np.append(digit_frequency, output)
    return output


# the FD is based on the following refrence
# Genton, Marc G. "The correlation structure of Matheron's classical variogram estimator under elliptically contoured distributions." Mathematical geology 32 (2000): 127-137.
# Tosi, Patrizia, et al. "Seismic signal detection by fractal dimension analysis." Bulletin of the Seismological Society of America 89.4 (1999): 970-977.
def calFD_feature(data, SPS, num=4):
    '''
    Parameters
    ----------
    data: data after dtrend+dmean, or raw data, np array
    SPS: sampling frequency of the seismic data, int
    num: how many data to select to fit the double-log curve

    Returns: Fractal diminasion, float
    -------
    '''

    # to store timeLag, tau, gamma_tau
    raw_array = np.empty((0, 3))
    # how many timeLag could be gerenated
    # because we only use 4 points to fit the curve, for minuing the calculation time,
    # here we only use 50
    max_timeLag = 50#len(data)-1

    # <editor-fold desc=" calculate the ">

    # timeLag = 1 is 1/SPS unit in real word
    for timeLag in range(1, max_timeLag):
        # make sure that each id in seq1 minus seq2 equals to timeLag
        sequence1 = data[ timeLag: ]
        sequence2 = data[ 0: len(data)-timeLag ]

        versusDistance = sequence1 - sequence2
        versusDistance = np.sum(versusDistance * versusDistance)

        Nh = len(sequence1) # len(sequence1) +  timeLag = max_timeLag
        tau = timeLag / SPS
        gamma_tau = versusDistance/ (2 * Nh)

        record = np.array([timeLag, tau, gamma_tau], dtype=float)
        raw_array = np.vstack((raw_array, record))
    # </editor-fold>


    # <editor-fold desc="fit the slope in double-log coordinates">
    x = raw_array[0:num, 1]
    x = np.log10(x)

    y = raw_array[0:num, 2]
    y = np.log10(y)

    z = np.polyfit(x, y, deg=1)

    FD = 2 - z[0] / 2
    if FD > 2:
        FD = 2

    FD = round(FD, 3)
    # </editor-fold>

    return FD
# </editor-fold>


# <editor-fold desc="**1** set input-output path">
def set_path (cal_year, input_station, input_component):
    global OUTPUT_DIR, SAC_DIR
    OUTPUT_DIR, SAC_DIR = "check-the-path", "check-the-path"
    if platform.system() == "Darwin":  # your local Mac PC name
        SAC_DIR, OUTPUT_DIR = "/Users/qizhou/Downloads/", "/Users/qizhou/Downloads/"
    elif platform.system() == "Linux":  # your remote server name
        SAC_DIR = f"/home/qizhou/0data/{cal_year}/{input_station}/{input_component}/"
        OUTPUT_DIR = f"/home/qizhou/1projects/dataForML/out60/{cal_year}/"
    return OUTPUT_DIR, SAC_DIR
# </editor-fold>


# <editor-fold desc="**2** read seismic data">
def get_seismicSignal(data_name):
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

#def cal_attributesBL(data_array): # the main function is from Qi
    #feature_array = calBL_feature(data_array)
    #return feature_array # 17 features

def record_data(cal_year, station, component, arr, RF_or_BL):
    arr = arr.reshape(1, -1)
    # seismic features to be saved
    if RF_or_BL == "RF":
        with open(f"{OUTPUT_DIR}{cal_year}_{station}_{component}_RF.txt", 'a') as file:
            np.savetxt(file, arr, header='', delimiter=',', comments='', fmt='%s')#'%.4f')
    elif RF_or_BL == "BL":
        with open(f"{OUTPUT_DIR}{cal_year}_{station}_{component}_BL.txt", 'a') as file:
            np.savetxt(file, arr, header='', delimiter=',', comments='', fmt='%s')#'%.4f')
    else:
        print("error at record_data")
# </editor-fold>


# <editor-fold desc="**4** run the calculator">
def cal_loop (cal_year, cal_window, station, component, id1, id2):

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
    # Mapping for ILL stations and components, default to -1 if not found station or component
    stationMapping = {'ILL18': 8.0, 'ILL12': 2.0, 'ILL13': 3.0, 'IGB01': 8.0, 'IGB02': 2.0, 'IGB10':3.0}
    componentMapping = {'EHZ': 1.0, 'EHE': 2.0, 'EHN': 3.0, 'BHZ': 1.0, 'BHE': 2.0, 'BHN': 3.0}

    # list all available sac file
    all_sacFile = os.listdir(SAC_DIR)

    # give features title and be careful the file name and path
    with open(f"{OUTPUT_DIR}{cal_year}_{station}_{component}_RF.txt", 'a') as file1:
        np.savetxt(file1, [featureName1], header='', delimiter=',', comments='', fmt='%s')
    # give features title and be careful the file name and path
    with open(f"{OUTPUT_DIR}{cal_year}_{station}_{component}_BL.txt", 'a') as file2:
        np.savetxt(file2, [featureName2], header='', delimiter=',', comments='', fmt='%s')


    for julday in range(id1, id2):  # 91 = 1st of May to 305=31 of Nov.
        julday = str(julday).zfill(3)
        miniSeed_name = f"{miniSEED}.{julday}"

        # columns array to store the seismic data for network features
        seismic_array = np.empty((0, SPS*cal_window+3))# creat the 3identifiers + data in cal_window*SPS

        d = UTCDateTime(year=cal_year, julday=julday)  # the start day, e.g.2014-07-12 00:00:00
        if miniSeed_name in all_sacFile:  # Yes, have seismic data in this julian day
            st, stEnvlope = get_seismicSignal(data_name=miniSeed_name)

            for step in range(0, int(60*60*24/cal_window)):  # 1440 = 24h * 60min
                d1 = d + (step) * cal_window     # from minute 0, 1, 2, 3, 4
                d2 = d + (step + 1) * cal_window # from minute 1, 2, 3, 4, 5

                # float timestamps, mapping station and component
                timeDate = datetime.fromtimestamp(d2.timestamp, tz=pytz.utc)
                timeDate = timeDate.strftime('%Y-%m-%d %H:%M:%S')
                id = np.array([timeDate, d2.timestamp, componentMapping.get(component, -1), stationMapping.get(station, -1)])

                tr1 = st.copy()
                tr2 = stEnvlope.copy()
                try:  # hava data in this time domain d1 to d2
                    tr1.trim(starttime=d1, endtime=d2, nearest_sample=False)
                    tr2.trim(starttime=d1, endtime=d2, nearest_sample=False)
                    # calculate the RF features
                    if len(tr1[0].data) >= SPS*cal_window : # full time window with 100data * 120s
                        seismic = tr1[0].data[:cal_window * SPS]
                        arr1 = cal_attributesRF(data_array=seismic)
                    else: # incomplete data ！= 100data * 120s
                        seismic, arr1 = np.full(cal_window * SPS, np.nan), np.full(59, np.nan)
                    # calculate the BL features
                    benford = calBL_feature(data=tr2[0].data, ruler=RULER)
                    # calculate the FD features
                    fd = calFD_feature(data=tr1[0].data, SPS=SPS)
                    BF_FD = np.append(benford, fd)
                except:  # NOT hava any data in this time domain d1 to d2
                    # calculate the RF features
                    seismic, arr1 = np.full(cal_window * SPS, np.nan), np.full(59, np.nan)
                    # calculate the BL features
                    benford = np.full(17, np.nan)
                    fd = np.full(1, np.nan)
                    BF_FD = np.append(benford, fd)

                arr_RF, arr_BL_FD = np.append(id, arr1), np.append(id, BF_FD)
                record_data(cal_year, station, component, arr_RF,    "RF") # write data by custom function
                record_data(cal_year, station, component, arr_BL_FD, "BL") # write data by custom function

                waveforms = np.append(id[1:].astype(float), seismic)
                seismic_array = np.vstack((seismic_array, waveforms))

        else:  # DoNOT, have any seismic data in this julian day
            for step in range(0, int(60*60*24/cal_window)):  # 1440 = 24h * 60min
                #d1 = d + (step) * cal_window     # from min 0, 1, 2, 3, 4
                d2 = d + (step + 1) * cal_window  # from min 1, 2, 3, 4, 5

                # float timestamps, mapping station and component
                timeDate = datetime.fromtimestamp(d2.timestamp, tz=pytz.utc)
                timeDate = timeDate.strftime('%Y-%m-%d %H:%M:%S')
                id = np.array([timeDate, d2.timestamp, componentMapping.get(component, -1), stationMapping.get(station, -1)])

                # calculate the RF features
                seismic, arr1 = np.full(cal_window * SPS, np.nan), np.full(59, np.nan)
                # calculate the BL features
                benford = np.full(17, np.nan)
                fd = np.full(1, np.nan)
                BF_FD = np.append(benford, fd)

                arr_RF, arr_BL_FD = np.append(id, arr1), np.append(id, BF_FD)
                record_data(cal_year, station, component, arr_RF,    "RF") # write data by custom function
                record_data(cal_year, station, component, arr_BL_FD, "BL")  # write data by custom function

                waveforms = np.append(id[1:].astype(float), seismic)
                seismic_array = np.vstack((seismic_array, waveforms))

        print(f"done: {miniSeed_name}")

        # save the npy every julday
        np.save(f"{OUTPUT_DIR}/{station}/{component}/{cal_year}_{station}_{component}_{julday}.npy", seismic_array)

# </editor-fold>

def main(year, input_station, input_component):  # Update the global variables with the values from command-line arguments
    print(f"Start Job: {year}, {input_station}: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S") )

    global SPS, OUTPUT_DIR, SAC_DIR, RULER, miniSEED
    '''
    #UTCDateTime("2017-05-17 00:00:00").julday, UTCDateTime("2018-05-15 00:00:00").julday
    #UTCDateTime("2019-05-15 00:00:00").julday, UTCDateTime("2020-05-01 00:00:00").julday
    #yearMapping1 = {2013: 147, 2014: 91, 2017: 137, 2018: 135, 2019: 135, 2020: 122}
    #id1 = yearMapping1.get(year, 91) #if not found, set as 91
    #UTCDateTime("2017-08-03 00:00:00").julday, UTCDateTime("2018-09-12 00:00:00").julday,
    #UTCDateTime("2019-10-23 00:00:00").julday, UTCDateTime("2020-10-01 00:00:00").julday
    #yearMapping2 = {2013: 245, 2014: 273, 2017: 215, 2018: 255, 2019: 296, 2020: 275}
    #id2 = yearMapping2.get(year, 305) #if not found, set as 305
    '''
    yearMappingStart = {2013:147, 2014:91, 2017:137, 2018:135, 2019:135, 2020:149}
    yearMappingEnd = {2013:245, 2014:273, 2017:215, 2018:255, 2019:291, 2020:275}
    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(job_id)
    if year in [2013, 2014]:  # 2013-2014 data
        OUTPUT_DIR, SAC_DIR = set_path(year, input_station, input_component)
        SPS = 200
        RULER = 1e3
        miniSEED = f"GM.{input_station}.{input_component}.{year}"
    elif year not in [2013, 2014]:  # 2017-2020 data
        OUTPUT_DIR, SAC_DIR = set_path(year, input_station, input_component)
        SPS = 100
        RULER = 1e2
        miniSEED = f"9S.{input_station}.{input_component}.{year}"
    else:
        print("error in OUTPUT_DIR, SAC_DIR = set_path")

    id1, id2 = yearMappingStart.get(year, 91),  yearMappingEnd.get(year, 300)
    # cal_loop (cal_year, cal_window, station, component, id1, id2)
    cal_loop(year, 60, input_station, input_component, id1=id1, id2=id2)



    print(f"End Job: {year}, {input_station}: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2020, help="check the year")
    parser.add_argument("--input_station", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_component", type=str, default="EHZ", help="check the input_station")

    args = parser.parse_args()
    main(args.year, args.input_station, args.input_component)

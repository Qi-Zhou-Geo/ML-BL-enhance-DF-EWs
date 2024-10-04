#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission


import os
import sys
import argparse
from datetime import datetime

import pytz

import numpy as np
import pandas as pd

from scipy.signal import hilbert, lfilter, butter, spectrogram
from scipy.stats import kurtosis, skew, iqr

from obspy import read, Stream, read_inventory, signal
from obspy.core import UTCDateTime # default is UTC+0 time zone
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('TkAgg')  # Or 'TkAgg'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print(parent_dir)

from calculate_features.Type_A_features import *      # import Qi's all features (by *)
from calculate_features.Type_B_features import *      # import Clement's all features (by *)

plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )


def cal_attributes_B(data_array, sps): # the main function is from Clement
    # sps: sampling frequency; flag=0: one component seismic signal;
    features = calculate_all_attributes(Data=data_array, sps=sps, flag=0)[0] # feature 1 to 60
    feature_array = features[1:]# leave features[0]=event_duration
    return feature_array # 59 features

def cal_attributes_A(data_array, ruler=300): # the main function is from Qi
    data_array_nm = data_array * 1e9 # converty m/s to nm/s
    feature_array = calBL_feature(data_array_nm, ruler)

    return feature_array # 17 features


def loop_time_step(data_start, data_duration, st, overlap, input_station="ILL12", input_component="EHZ", input_window_size=60):
    total_step = int((3600 * data_duration) / (input_window_size * overlap))
    # columns array to store the seismic data for network features
    sps = int(st[0].stats.sampling_rate)
    d = UTCDateTime(data_start)  # the start day, e.g.2014-07-12 00:00:00
    arr_RF, arr_BL = np.empty((1, 63)), np.empty((1, 21))

    for step in range(0, total_step):  # 1440 = 24h * 60min
        d1 = d + step * input_window_size * overlap                # from minute 0, 1, 2, 3, 4
        d2 = d1 + input_window_size  # from minute 2, 3, 4, 5, 6
        if step < 5:
            print(data_start, data_duration, overlap, d1, d2)
        # float timestamps, mapping station and component
        time = datetime.fromtimestamp(d1.timestamp, tz=pytz.utc)
        time = time.strftime('%Y-%m-%d %H:%M:%S')
        id = np.array([time, d1.timestamp, input_station, input_component])

        tr = st.copy()
        tr.trim(starttime=d1, endtime=d2, nearest_sample=False)
        seismic_data = tr[0].data[:sps * input_window_size]
        type_B_arr = cal_attributes_B(data_array=seismic_data, sps=sps)
        type_A_arr = cal_attributes_A(data_array=seismic_data)

        arr_RF_temp, arr_BL_temp = np.append(id, type_B_arr), np.append(id, type_A_arr)
        arr_RF = np.vstack((arr_RF, arr_RF_temp))
        arr_BL = np.vstack((arr_BL, arr_BL_temp))

    return arr_RF, arr_BL


data_start = "2020-06-29 01:00:00"
data_duration = 8
st =read('/Users/qizhou/#SAC/2017-2020/9S.ILL13.EHZ.2020.181')
st.merge(method=1, fill_value='latest', interpolation_samples=0)
st._cleanup()
st.detrend('linear')
st.detrend('demean')
inv = read_inventory(f"/Users/qizhou/#SAC/2017-2020/metadata_2017-2020.xml")
st.remove_response(inventory=inv)
st.filter("bandpass", freqmin=1, freqmax=45)
st = st.trim(starttime=UTCDateTime(data_start),
             endtime=UTCDateTime(data_start) + data_duration * 3600 + 60, nearest_sample=False)
st.detrend('linear')
st.detrend('demean')



# no ovelap
arr_RF, arr_BL = loop_time_step(data_start, data_duration, st, overlap=1)
# 50% (30 seconds) ovelap
arr_RF_30, arr_BL_30 = loop_time_step(data_start, data_duration, st, overlap=0.5)



fig = plt.figure(figsize=(5.5, 4))
gs = gridspec.GridSpec(2, 1)
scaler = MinMaxScaler()

ax = plt.subplot(gs[0])
es15_25 = arr_RF[1:, 17].astype(float) # 'ES_2' ES 15-25
alpha = arr_BL[1:, 17].astype(float) # power law exponent
alpha[alpha > 5] = 5
es15_25 = scaler.fit_transform(es15_25.reshape(-1, 1)).reshape(-1)
alpha = scaler.fit_transform(alpha.reshape(-1, 1)).reshape(-1)

plt.plot(es15_25, label = "ES 15-25 Hz", zorder=2)
plt.plot(alpha, label = "Power law exponent", zorder=2)
plt.legend(loc="center left", fontsize=6)
plt.xlim(0, arr_BL.shape[0])
plt.title("Window size = 60s, overlap=0", loc="left", fontsize=6)
plt.grid(axis='y', ls="--", lw=0.5, zorder=1)


xLocation = np.arange(0, arr_BL.shape[0], 120)
xTicks = [(UTCDateTime(data_start) + i * 60).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
plt.xticks(xLocation, xTicks)
plt.xlabel("UTC+0 Time", weight="bold")
plt.ylabel("Normalized Value", weight="bold")

ax = plt.subplot(gs[1])
es15_25 = arr_RF_30[1:, 17].astype(float) # 'ES_2' ES 15-25
alpha = arr_BL_30[1:, 17].astype(float) # power law exponent
alpha[alpha > 5] = 5
es15_25 = scaler.fit_transform(es15_25.reshape(-1, 1)).reshape(-1)
alpha = scaler.fit_transform(alpha.reshape(-1, 1)).reshape(-1)

plt.plot(es15_25, label = "ES 15-25 Hz", zorder=2)
plt.plot(alpha, label = "Power law exponent", zorder=2)
plt.legend(loc="center left", fontsize=6)
plt.xlim(0, arr_BL_30.shape[0])
plt.title("Window size = 60s, overlap=30s (50%)", loc="left", fontsize=6)
plt.grid(axis='y', ls="--", lw=0.5, zorder=1)

xLocation = np.arange(0, arr_BL_30.shape[0], 240)
xTicks = [(UTCDateTime(data_start) + i * 60 * 0.5).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
plt.xticks(xLocation, xTicks)
plt.xlabel("UTC+0 Time", weight="bold")
plt.ylabel("Normalized Value", weight="bold")


plt.tight_layout()
plt.savefig(f"test_overlap_window.png", dpi=600)
plt.show()


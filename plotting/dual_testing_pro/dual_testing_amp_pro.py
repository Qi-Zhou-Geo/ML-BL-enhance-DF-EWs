#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
import os
import sys

import pytz
import numpy as np
import pandas as pd
from datetime import datetime
from obspy import read, Stream, UTCDateTime, read_inventory

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Get the absolute path of the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# internal functions
from config.config_dir import CONFIG_dir
from calculate_features.seismic_data_processing import *
from calculate_features.remove_sensor_response import *
from plotting.dual_testing_pro.fetch_predicted_pro import fetc_data


plt.rcParams.update( {'font.size':7} )#, 'font.family': "Arial"} )

def increased_warning_time(data_start, st):
    max_imp = np.where(st[0].data == np.max(st[0].data))[0][0]

    max_imp_time = UTCDateTime(data_start) + max_imp / st[0].stats.sampling_rate
    max_imp_time = max_imp_time.strftime('%Y-%m-%d %H:%M:%S')

    return max_imp_time


def plot_func(seismic_network, input_station, input_component, data_start, data_end):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.2, 1, 1, 1, 1])

    ax = plt.subplot(gs[0])
    st = load_seismic_signal(seismic_network, input_station, input_component, data_start, data_end)
    st.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
    ax.images[0].set_clim(-180, -100)
    plt.ylabel("Frequency (Hz)", fontweight='bold')
    plt.yticks([1, 10, 20, 30, 40, 45], [1, 10, 20, 30, 40, 45])
    plt.ylim(1, 45)
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval))  # unit is second

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="7%", pad=0.05)
    cbar = plt.colorbar(mappable=ax.images[-1], cax=cax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')  # Move color bar tick labels to the top
    cbar.set_label('Power Spectral Density PSD (dB)', fontsize=6, labelpad=-25, fontweight="bold")

    ax = plt.subplot(gs[1])
    ax.plot(st[0].data, lw=1, label=f"{seismic_network}-{input_station}-{input_component}", color="black")
    plt.xlim(0, st[0].data.size - 1)
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval * st[0].stats.sampling_rate))  # unit is second
    plt.ylabel("Amplitude (m/s)", fontweight='bold')
    plt.legend(loc="upper right", fontsize=6)
    plt.grid(axis='y', ls="--", lw=0.5, zorder=1)

    for idx, feature_type in enumerate(["A", "B", "C"]):
        ax = plt.subplot(gs[idx + 2])

        for model_type in ["Random_Forest", "XGBoost", "LSTM"]:
            pro, false_positive = fetc_data(data_path, input_station, model_type, feature_type, input_component, data_start, data_end)
            ax.plot(pro, lw=1, label=f"{model_type}-{feature_type}, FP={false_positive}", zorder=2)
            plt.xlim(0, len(pro))

        plt.ylabel("Predicted\nDF Pro", fontweight='bold')
        plt.legend(loc="upper left", fontsize=6)
        plt.grid(axis='y', ls="--", lw=0.5, zorder=1)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(60 * x_interval))  # unit is second

        if idx != 2:
            ax.axes.xaxis.set_ticklabels([])
        else:
            plt.xlabel(f"UTC+0", fontweight='bold')

    duration = int((UTCDateTime(data_end) - UTCDateTime(data_start)) / 3600)
    xLocation = np.arange(0, 60 * (duration + x_interval), 60 * x_interval)
    xTicks = [(UTCDateTime(data_start) + i * 60).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
    plt.xticks(xLocation, xTicks)

    plt.tight_layout()
    plt.savefig(f"{CONFIG_dir['parent_dir']}/plotting/dual_testing_pro/"
                f"{seismic_network}_{input_station}_{input_component}_{data_start}_{data_end}.png", dpi=600)


# meta information, time is UTC+0
seismic_network, input_station, input_component = "1A", "E19A", "CHZ"
data_path = f"{CONFIG_dir['output_dir']}/dual_test_{seismic_network}/"
x_interval = 0.5 # unit is hour

time1 = ["2021-07-16 19:00:00", "2021-08-17 18:00:00", "2021-07-14 19:30:00"]
time2 = ["2021-07-16 22:00:00", "2021-08-17 21:00:00", "2021-07-14 22:30:00"]

for step in np.arange(len(time1)):
    data_start, data_end = time1[step], time2[step]
    plot_func(seismic_network, input_station, input_component, data_start, data_end)



df = pd.read_csv('/Users/qizhou/Desktop/2022_ILL12_EHZ_all_A.txt', header=0)
df1 = df.iloc[:, :2]
df1['label_0nonDF_1DF'] = 0
date = np.array(df1.iloc[:, 0])

id1 = np.where(date == "2022-06-05 10:00:00")[0][0]
id2 = np.where(date == "2022-06-05 14:00:00")[0][0] + 1
df1.iloc[id1:id2, -1] = 1

id1 = np.where(date == "2022-06-30 19:00:00")[0][0]
id2 = np.where(date == "2022-06-30 22:00:00")[0][0] + 1
df1.iloc[id1:id2, -1] = 1

id1 = np.where(date == "2022-07-04 19:30:00")[0][0]
id2 = np.where(date == "2022-07-04 23:30:00")[0][0] + 1
df1.iloc[id1:id2, -1] = 1

id1 = np.where(date == "2022-09-08 00:00:00")[0][0]
id2 = np.where(date == "2022-09-08 03:00:00")[0][0] + 1
df1.iloc[id1:id2, -1] = 1


df1.to_csv('/Users/qizhou/Desktop/2022_ILL12_EHZ_observed_label.txt', sep=',', index=False, mode='w')


fig = plt.figure(figsize=(5.5, 3))
gs = gridspec.GridSpec(1, 1)

x = np.arange(361)
for idx, station in enumerate(["ILL12"]):
    ax = plt.subplot(gs[idx])
    for model_type in ["Random_Forest"]:
        for feature_type in ["A", "B", "C"]:
            df = pd.read_csv(f'/Users/qizhou/#python/#GitHub_saved/ML-BL-enhance-DF-EWs/output/dual_test_9S/'
                             f'{station}_{model_type}_{feature_type}_EHZ_dual_testing_output.txt', header=0)
            date = np.array(df1.iloc[:, 0])
            id1 = np.where(date == "2022-07-04 19:30:00")[0][0]
            id2 = np.where(date == "2022-07-05 01:30:00")[0][0] + 1
            pro = df.iloc[id1:id2, -1]

            ax.plot(x, pro, label=f"{station}-{model_type}-{feature_type}")
    #plt.legend(fontsize=6)
    plt.xlabel("UTC+0 Time", weight="bold")
    plt.ylabel("Probability", weight="bold")



plt.tight_layout()
plt.savefig(f"/Users/qizhou/Desktop/2022.png", dpi=600)
plt.show()

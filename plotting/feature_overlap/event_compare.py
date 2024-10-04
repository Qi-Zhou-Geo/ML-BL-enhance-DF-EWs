#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

parent_dir = os.path.dirname(__file__)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def increased_warning_time(st):
    max_imp = np.where(st[0].data == np.max(st[0].data))[0][0]

    max_imp_time = UTCDateTime(st[0].stats.starttime) + max_imp / st[0].stats.sampling_rate
    max_imp_time = max_imp_time.strftime('%Y-%m-%d %H:%M:%S')

    return max_imp_time



time1 = ["2014-07-12 13:00:00", "2020-06-29 03:00:00", "2021-08-17 18:00:00"]
time2 = ["2014-07-12 18:00:00", "2020-06-29 07:00:00", "2021-08-17 21:00:00"]
seismic_network = ["9J", "9S", "1A"]
input_station = ["IGB02", "ILL12", "E19A"]
input_component = ["HHZ", "EHZ", "CHZ"]
usage_type = ["training", "testing", "dual_testing"]
input_data_year = [[2017, 2018, 2019], [2014], [2021]]

a = 1
if a == 0:
    from obspy import read, Stream, UTCDateTime, read_inventory
    from calculate_features.seismic_data_processing import *  # load and process the seismic signals
    max_imp_time_list = []
    for step in np.arange(len(time1)):
        data_start, data_end = time1[step], time2[step]
        st = load_seismic_signal(seismic_network[step], input_station[step], input_component[step], data_start,
                                 data_end)
        max_imp_time = increased_warning_time(st)
        print(max_imp_time)
        max_imp_time_list.append(max_imp_time)


from sklearn.preprocessing import MinMaxScaler
from functions.dataset2dataloader import *

max_imp_time_list = ["2014-07-12 15:00:00", "2020-06-29 05:02:00", "2021-08-17 19:15:00"]
feature_type = "C"
feature_id = 17#35
temp = []
for step in np.arange(len(max_imp_time_list)):
    #input_features_name, X_train, _, _, time_stamps_test = #select_features(input_station[step],
                                                                           #feature_type,
                                                                           #input_component[step],
                                                                           #usage_type[step],
                                                                           #input_data_year[step])
    t = UTCDateTime(max_imp_time_list[step])
    df = pd.read_csv(f'/Users/qizhou/#file/2_projects/Luding/1figure/event_compare/'
                     f'{t.year}_{input_station[step]}_{input_component[step]}_{t.julday}_A.txt', header=0)
    time_stamps_test = np.array(df.iloc[:, 0])

    time_stamps_test = np.array(time_stamps_test)
    id1 = np.where(time_stamps_test == time1[step])[0][0]
    id2 = np.where(time_stamps_test == max_imp_time_list[step])[0][0]
    id3 = np.where(time_stamps_test == time2[step])

    data = np.array(df.iloc[(id2-60):(id2+120), feature_id]).reshape(-1, 1)
    data[data>5] = 10
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data).reshape(-1)
    temp.append(data)


fig = plt.figure(figsize=(5.5, 3))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

for step in np.arange(len(max_imp_time_list)):
    ax.plot(temp[step], label=f"{seismic_network[step]}-{max_imp_time_list[step]}", zorder=2)

plt.xlabel("Time", weight="bold")
plt.ylabel("Normalized Power law exponent", weight="bold")
plt.legend(loc="upper left", fontsize=6)

plt.tight_layout()
plt.savefig(f"/Users/qizhou/#file/2_projects/Luding/1figure/event_compare/Normalized Power law exponent.png", dpi=600)
plt.show()

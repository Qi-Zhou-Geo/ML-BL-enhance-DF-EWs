#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import pandas as pd
import numpy as np
import os
import pytz
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from obspy import UTCDateTime
import matplotlib.gridspec as gridspec

plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )#, 'font.weight':'bold'

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # get the parent path
df = pd.read_csv(f"{parent_dir}/plotting/network_warning/warning_summary.txt", header=None)
arr = np.array(df)

model_type_arr, feature_type_arr = arr[:, 0], arr[:, 1]
model_type_arr, feature_type_arr = np.char.strip(model_type_arr.astype(str)),np.char.strip(feature_type_arr.astype(str))


for model_type in ["Random_Forest", "XGBoost", "LSTM"]:
    for feature_type in ["A", "B", "C"]:

        id1 = np.where(model_type_arr == model_type)
        id2 = np.where(feature_type_arr == feature_type)

        intersection = np.intersect1d(id1, id2)
        intersection_arr = arr[intersection, :]

        pro_threshold = intersection_arr[:, 3] # always is 0 now 2024-09-05
        warning_threshold = intersection_arr[:, 4]
        attention_window_size = intersection_arr[:, 5]

        total_increase_warning = intersection_arr[:, 7] / 60 # original unit is second
        mean_increase_warning = total_increase_warning / 12
        max_increase_warning = intersection_arr[:, 8] / 60 # original unit is second
        min_increase_warning = intersection_arr[:, 9] / 60 # original unit is second
        failed_warning = intersection_arr[:, 10]

        # Create a DataFrame from the arrays
        df1 = pd.DataFrame({
            'x': attention_window_size.astype(float),
            'y': np.round(warning_threshold.astype(float), 1),
            'value': total_increase_warning.astype(float)
        })
        pivot_table1 = df1.pivot(index='y', columns='x', values='value')

        df2 = pd.DataFrame({
            'x': attention_window_size.astype(float),
            'y': np.round(warning_threshold.astype(float), 1),
            'value': failed_warning.astype(float)
        })
        pivot_table2 = df2.pivot(index='y', columns='x', values='value')


        sns.heatmap(pivot_table1, annot=pivot_table2)

        plt.xlabel("attention_window_size (minute)")
        plt.ylabel("warning_threshold")
        plt.title(f"{model_type}-{feature_type}")
        plt.tight_layout()
        plt.savefig(f"{parent_dir}/plotting/network_warning/{model_type}-{feature_type}.png", dpi=600)
        plt.show()




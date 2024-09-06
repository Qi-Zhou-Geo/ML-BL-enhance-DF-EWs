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
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )#, 'font.weight':'bold'
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # get the parent path


def create_discrate_cmap(total_num, max_value, interval=10):
    cmap = sns.color_palette('Oranges', total_num)  # Get 10 shades of blue
    cmap = mcolors.ListedColormap(cmap)  # Create a discrete colormap

    bounds = np.linspace(0, max_value, interval+1)  # 10 intervals between 0 and 150
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, bounds, norm


def convert2table(x, y, value):
    df = pd.DataFrame({
        'x': x.astype(float),
        'y': np.round(y.astype(float),1),
        'value': value.astype(float)
    })
    pivot_table = df.pivot(index='y', columns='x', values='value')

    return np.around(np.array(pivot_table))


def fetch_data(model_type, feature_type, data_type):

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # get the parent path
    df = pd.read_csv(f"{parent_dir}/plotting/network_warning/warning_summary.txt", header=None)
    arr = np.array(df)

    model_type_arr, feature_type_arr = arr[:, 0], arr[:, 1]
    model_type_arr = np.char.strip(model_type_arr.astype(str))
    feature_type_arr = np.char.strip(feature_type_arr.astype(str))

    id1 = np.where(model_type_arr == model_type)
    id2 = np.where(feature_type_arr == feature_type)

    intersection = np.intersect1d(id1, id2)
    intersection_arr = arr[intersection, :]

    pro_threshold = intersection_arr[:, 3]  # always is 0 now 2024-09-05
    warning_threshold = intersection_arr[:, 4]
    attention_window_size = intersection_arr[:, 5]

    total_increase_warning = intersection_arr[:, 9] / 60  # original unit is second
    mean_increase_warning = total_increase_warning / 12
    max_increase_warning = intersection_arr[:, 10] / 60  # original unit is second
    min_increase_warning = intersection_arr[:, 11] / 60  # original unit is second

    failed_warning_times = intersection_arr[:, 12]
    fasle_warning_times = intersection_arr[:, 7]

    if data_type == "failed_warning":
        try:
            id3 = np.where((failed_warning_times + fasle_warning_times) == 0)  # no false warning, no failured
            l1 = np.where(mean_increase_warning == np.max(mean_increase_warning[id3]))
            l2 = np.where(min_increase_warning == np.max(min_increase_warning[id3]))
            l3 = np.intersect1d(l1, l2)
            for idx, step in enumerate(l3):
                record = f"{model_type, feature_type, data_type}, " \
                         f"{idx + 1}/{len(l3)}, " \
                         f"global pro: {intersection_arr[step, 4]}, " \
                         f"window size: {intersection_arr[step, 5]}, " \
                         f"total increased: {intersection_arr[step, 9] / 60}, " \
                         f"min increased: {intersection_arr[step, 11] / 60}"
                print(record)

        except Exception as e:
            print(f"{model_type, feature_type, data_type}, {e}")
    else:
        pass


    if data_type == "min":
        pivot_table = convert2table(attention_window_size, warning_threshold, min_increase_warning)
    elif data_type == "mean":
        pivot_table = convert2table(attention_window_size, warning_threshold, mean_increase_warning)
    elif data_type == "failed_warning":
        pivot_table = convert2table(attention_window_size, warning_threshold, failed_warning_times)
    elif data_type == "fasle_warning_times":
        pivot_table = convert2table(attention_window_size, warning_threshold, fasle_warning_times)
    else:
        print(f"check the data type {data_type}")

    return pivot_table


def plot_network_warning(model_type, feature_type):

    fig = plt.figure(figsize=(5.5, 3))
    gs = gridspec.GridSpec(1, 2)

    ax = plt.subplot(gs[0])
    pivot_table = fetch_data(model_type, feature_type, "failed_warning")
    cmap, bounds, norm = create_discrate_cmap(10, 150)
    ax = sns.heatmap(pivot_table, annot=pivot_table, cmap='Oranges', cbar=False)

    plt.xticks([0.5, 4.5, 9.5, 14.5, 19.5], [1, 5, 10, 15, 20])
    plt.yticks(np.arange(0.5, 10, 1), np.round(np.arange(0.1, 1.1, 0.1), 1), horizontalalignment="right", rotation=0)

    plt.xlabel(r"Attention window size $l$ (minute)")
    plt.ylabel(r"Warning Threshold $Pr_{g}$")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.4)
    cbar = plt.colorbar(mappable=ax.get_children()[0], cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.set_label("Failured Warning Event", labelpad=6)

    ax = plt.subplot(gs[1])
    pivot_table = fetch_data(model_type, feature_type, "fasle_warning_times")

    print(np.max(np.array(pivot_table)))
    cmap = sns.color_palette('Oranges', 6)
    cmap = mcolors.ListedColormap(cmap)
    bounds = np.array([0, 5, 10, 100, 200, 1100])  # np.linspace(0, 1100, 5)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    label = np.where(pivot_table > 10, "", pivot_table.astype(int))

    ax = sns.heatmap(pivot_table, annot=label, cmap=cmap, norm=norm, cbar=False, fmt='', cbar_kws={"ticks": bounds})

    plt.xticks([0.5, 4.5, 9.5, 14.5, 19.5], [1, 5, 10, 15, 20])
    plt.yticks(np.arange(0.5, 10, 1), np.round(np.arange(0.1, 1.1, 0.1), 1), horizontalalignment="right", rotation=0)

    plt.xlabel(r"Attention window size $l$ (minute)")
    plt.ylabel("")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.4)
    cbar = plt.colorbar(mappable=ax.get_children()[0], cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.set_label("Failured Warning Event", labelpad=6)

    plt.tight_layout()
    plt.savefig(f"{parent_dir}/plotting/network_warning/fig/{model_type}-{feature_type}.png", dpi=600)
    plt.show()



for model_type in ["Random_Forest", "XGBoost", "LSTM"]:
    for feature_type in ["A", "B", "C"]:
        plot_network_warning(model_type, feature_type)

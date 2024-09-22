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

    total_increase_warning = intersection_arr[:, 9].astype(float) / 60  # original unit is second
    mean_increase_warning = total_increase_warning.astype(float) / 12
    max_increase_warning = intersection_arr[:, 10].astype(float) / 60  # original unit is second
    min_increase_warning = intersection_arr[:, 11].astype(float) / 60  # original unit is second

    failed_warning_times = intersection_arr[:, 12].astype(float)
    fasle_warning_times = intersection_arr[:, 7].astype(float)

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


def plot_network_warning1(ax0, ax1, model_type, feature_type, legend_label):

    pivot_table = fetch_data(model_type, feature_type, "failed_warning")
    label = np.where(pivot_table > 0, "", pivot_table.astype(int))

    cmap = sns.color_palette('Oranges', 13)
    cmap = mcolors.ListedColormap(cmap)
    bounds = np.arange(0, 13)#np.array([0, 5, 10, 100, 200, 1100])  # np.linspace(0, 1100, 5)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    print(np.max(pivot_table))

    sns.heatmap(pivot_table, annot=label, cmap=cmap, norm=norm, cbar=False, fmt='', cbar_kws={"ticks": bounds}, ax=ax0)

    ax0.set_xticks([0.5, 4.5, 9.5, 14.5, 19.5], [1, 5, 10, 15, 20])
    ax0.set_yticks(np.arange(0.5, 10, 1), np.round(np.arange(0.1, 1.1, 0.1), 1), horizontalalignment="right", rotation=0)

    if legend_label is True:
        #plt.xlabel(r"Attention Window Size (minute)")
        #plt.ylabel(r"Global Averaged Probability")

        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("bottom", size="10%", pad=0.5)
        cbar = plt.colorbar(mappable=ax0.get_children()[0], cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_label("Number of Failed Warning Events", labelpad=6)
    else:
        pass


    pivot_table = fetch_data(model_type, feature_type, "fasle_warning_times")
    #label = np.where((pivot_table <= 0) | (pivot_table > 100), "", pivot_table.astype(int))
    label = np.where(pivot_table > 0, "", pivot_table.astype(int))


    cmap = sns.color_palette('Reds', 7)
    cmap = mcolors.ListedColormap(cmap)
    bounds = np.array([0, 1, 5, 10, 50, 100, 200])#np.array([0, 5, 10, 100, 200, 1100])  # np.linspace(0, 1100, 5)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    print(np.max(pivot_table))

    sns.heatmap(pivot_table, annot=label, cmap=cmap, norm=norm, cbar=False, fmt='', cbar_kws={"ticks": bounds}, ax=ax1)

    ax1.set_xticks([0.5, 4.5, 9.5, 14.5, 19.5], [1, 5, 10, 15, 20])
    ax1.set_yticks(np.arange(0.5, 10, 1), np.round(np.arange(0.1, 1.1, 0.1), 1), horizontalalignment="right", rotation=0)

    if legend_label is True:
        #plt.xlabel(r"Attention Window Size (minute)")
        #plt.ylabel("")

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="10%", pad=0.5)
        cbar = plt.colorbar(mappable=ax1.get_children()[0], cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_ticks([0, 1, 5, 10, 50, 100, 200])
        cbar.set_ticklabels([0, 1, 5, 10, 50, 100, ">100"])
        cbar.set_label("Number of False Warnings", labelpad=6)
    else:
        pass

def plot_network_warning(ax0, ax1, model_type, feature_type, legend_label):

    pivot_table = fetch_data(model_type, feature_type, "mean")
    print(np.max(pivot_table))

    sns.heatmap(pivot_table, cmap='Oranges', vmin=0, vmax=100, ax=ax0)

    ax0.set_xticks([0.5, 4.5, 9.5, 14.5, 19.5], [1, 5, 10, 15, 20])
    ax0.set_yticks(np.arange(0.5, 10, 1), np.round(np.arange(0.1, 1.1, 0.1), 1), horizontalalignment="right", rotation=0)

    if legend_label is True:
        #plt.xlabel(r"Attention Window Size (minute)")
        #plt.ylabel(r"Global Averaged Probability")

        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("bottom", size="10%", pad=0.5)
        cbar = plt.colorbar(mappable=ax0.get_children()[0], cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_label("Number of Failed Warning Events", labelpad=6)
    else:
        pass


    pivot_table = fetch_data(model_type, feature_type, "min")
    print(np.max(pivot_table))

    sns.heatmap(pivot_table, cmap='Greens', vmin=0, vmax=60, ax=ax1)

    ax1.set_xticks([0.5, 4.5, 9.5, 14.5, 19.5], [1, 5, 10, 15, 20])
    ax1.set_yticks(np.arange(0.5, 10, 1), np.round(np.arange(0.1, 1.1, 0.1), 1), horizontalalignment="right", rotation=0)

    if legend_label is True:
        #plt.xlabel(r"Attention Window Size (minute)")
        #plt.ylabel("")

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("bottom", size="10%", pad=0.5)
        cbar = plt.colorbar(mappable=ax1.get_children()[0], cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.set_ticks([0, 1, 5, 10, 50, 100, 200])
        cbar.set_ticklabels([0, 1, 5, 10, 50, 100, ">100"])
        cbar.set_label("Number of False Warnings", labelpad=6)
    else:
        pass


feature_type = "C"
fig = plt.figure(figsize=(4.5, 5.5))
gs = gridspec.GridSpec(3, 2)

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax0.set_title(f"(a) Random Forest Model with Type {feature_type}", loc='left', fontsize=6, fontweight='bold')
plot_network_warning(ax0, ax1, "Random_Forest", feature_type, False)

ax0 = plt.subplot(gs[2])
ax1 = plt.subplot(gs[3])
ax0.set_title(f"(b) XGBoost Model with Type {feature_type}", loc='left', fontsize=6, fontweight='bold')
plot_network_warning(ax0, ax1, "XGBoost", feature_type, False)


ax0 = plt.subplot(gs[4])
ax1 = plt.subplot(gs[5])
ax0.set_title(f"(c) LSTM Model with Type {feature_type}", loc='left', fontsize=6, fontweight='bold')
plot_network_warning(ax0, ax1, "LSTM", feature_type, False)


fig.text(x=0, y=0.5, s="Global Averaged Probability", fontsize=7, weight="bold", va='center', rotation='vertical')
fig.text(x=0.4, y=0.01, s="Attention Window Size (minute)", fontsize=7, weight="bold", va='center')

plt.tight_layout()
plt.savefig(f"{parent_dir}/plotting/network_warning/fig/warning_strategy_matrix_{feature_type}_time.png", dpi=600)
plt.show()

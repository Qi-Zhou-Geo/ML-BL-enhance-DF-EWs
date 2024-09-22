#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
import os
import pytz
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )
parent_dir = os.path.dirname(os.path.abspath(__file__))  # get the parent path


def gosia_warning():
    df1 = pd.read_csv(f"{parent_dir}/2020_Gosia_warning.txt", header=0)
    df2 =  pd.read_csv(f"{parent_dir}/2020_CD29time.txt", header=None)

    pre_cd1 = []
    post_cd1 = []

    for idx in range(len(df1)):
        pre = df1.iloc[idx, 1]
        post = df1.iloc[idx, 4]
        cd29 = df2.iloc[idx, 0]

        if pre is np.nan or post is np.nan :
            if pre is np.nan:
                pre_warning = np.nan
            else:
                post_warning = np.nan
        else:
            pre = datetime.strptime(pre, "%Y-%m-%d %H:%M:%S")
            post = datetime.strptime(post, "%Y-%m-%d %H:%M:%S")
            cd29 = datetime.strptime(cd29, "%Y-%m-%d %H:%M:%S")

            pre_warning = (cd29 - pre).total_seconds()  # in second
            post_warning = (cd29 - post).total_seconds()  # in second

        pre_cd1.append(pre_warning)
        post_cd1.append(post_warning)

    return  np.array(pre_cd1)/60, np.array(post_cd1)/60


def fetch_data(model_type, feature_type, data_type, pro_threshold, warning_threshold, attention_window_size):

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # get the parent path
    df = pd.read_csv(f"{parent_dir}/plotting/network_warning/warning_summary.txt", header=None)
    arr = np.array(df)

    model_type_arr, feature_type_arr = arr[:, 0], arr[:, 1]
    model_type_arr = np.char.strip(model_type_arr.astype(str))
    feature_type_arr = np.char.strip(feature_type_arr.astype(str))

    id1 = np.where(model_type_arr == model_type)
    id2 = np.where(feature_type_arr == feature_type)
    id3 = np.where(arr[:, 3] == pro_threshold)
    id4 = np.where(arr[:, 4] == warning_threshold)
    id5 = np.where(arr[:, 5] == attention_window_size)

    arrays = [id1, id2, id3, id4, id5]
    intersection = arrays[0]
    for arr_step in arrays[1:]:
        intersection = np.intersect1d(intersection, arr_step)

    intersection_arr = arr[intersection[0], :]


    return intersection_arr[13:].astype(float)


def plot_bar_gosia(ax, pre_cd1, post_cd1, idx):

    bar_width = 0.35
    x = np.arange(pre_cd1.size)
    x1 = [x - bar_width / 2 for x in x]
    x2 = [x + bar_width / 2 for x in x]

    for step in range(pre_cd1.size):
        if np.isnan(pre_cd1[step]):
            ax.bar(x1[step], 300, width=bar_width, edgecolor='grey', label="pre-cd1", color='grey', alpha=0.3, zorder=2)
        else:
            ax.bar(x1[step], pre_cd1[step], width=bar_width, edgecolor='grey', label="no data", color='green', zorder=2)

    for step in range(post_cd1.size):
        if np.isnan(post_cd1[step]):
            ax.bar(x2[step], 300, width=bar_width, edgecolor='grey', label="post-cd1", color='grey', alpha=0.3, zorder=2)
        else:
            ax.bar(x2[step], post_cd1[step], width=bar_width, edgecolor='grey', label="no data", color='orange', zorder=2)

    plt.grid(axis='y', ls="--", lw=0.5, zorder=1)

    plt.xticks(x, ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7a', 'E7b', 'E8', 'E9', 'E10', 'E11'])
    plt.ylabel('Increased Warning Time\n(minute)', fontweight='bold')


    custom_legend = [Patch(edgecolor='grey', label="Pre-CD1 Warning", facecolor='green'),
                     Patch(edgecolor='grey', label="Post-CD1 Warning", facecolor='orange'),
                     Patch(edgecolor='grey', label="Failured Warning", facecolor='grey', alpha=0.3)]

    ax.legend(handles=custom_legend, loc="upper right")

    plt.ylim(0, 240)
    plt.xlim(min(x1) - bar_width, max(x2) + bar_width)
    plt.text(x=min(x1) - bar_width, y=200, s=f' ({idx})', weight="bold")



def plot_bar(ax, model1, model2, model3, pre_cd1, post_cd1, idx, faliured=False):

    bar_width = 0.25

    x1 = np.arange(12)  # Positions for model1
    x2 = [x + bar_width for x in x1]  # Positions for model2
    x3 = [x + bar_width for x in x2]  # Positions for warning

    print(x1, x2, x3)
    for step in range(model1.size):
        if np.isnan(model1[step]) or model1[step] == 0:
            ax.bar(x1[step], 300, width=bar_width, edgecolor='grey', color='grey', alpha=0.3, zorder=2)
        else:
            ax.bar(x1[step], model1[step], width=bar_width, edgecolor='grey', label="no data", color='green', zorder=2)

    for step in range(model2.size):
        if np.isnan(model2[step]) or model2[step] == 0:
            ax.bar(x2[step], 300, width=bar_width, edgecolor='grey', color='grey', alpha=0.3, zorder=2)
        else:
            ax.bar(x2[step], model2[step], width=bar_width, edgecolor='grey', label="no data", color='orange', zorder=2)

    for step in range(model3.size):
        if np.isnan(model3[step]) or model3[step] == 0:
            ax.bar(x3[step], 300, width=bar_width, edgecolor='grey', color='grey', alpha=0.3, zorder=2)
        else:
            ax.bar(x3[step], model3[step], width=bar_width, edgecolor='grey', label="no data", color='blue', zorder=2)


    for step in range(pre_cd1.size):
        if np.isnan(pre_cd1[step]):
            benckmark_result(200, x1[step], x3[step], "pre-cd1", 'grey')
        else:
            benckmark_result(pre_cd1[step], x1[step], x3[step], "pre-cd1", "red")

    for step in range(post_cd1.size):
        if np.isnan(post_cd1[step]):
            benckmark_result(200, x1[step], x3[step], "post-cd1", 'grey')
        else:
            benckmark_result(post_cd1[step], x1[step], x3[step], "post-cd1", "black")



    if faliured == True:
        custom_legend = [Patch(edgecolor='grey', label="RF", facecolor='green'),
                         Patch(edgecolor='grey', label="XGBoost", facecolor='orange'),
                         Patch(edgecolor='grey', label="LSTM", facecolor='blue'),
                         Line2D([0], [0], color='red', lw=1.5, label="Pre-CD1"),
                         Line2D([0], [0], color='black', lw=1.5, label="Post-CD1"),
                         Line2D([0], [0], color='grey', lw=1.5, label="Failured")]
        ax.legend(handles=custom_legend, bbox_to_anchor=(0.62, 0.6), fontsize=6, ncol=2)
    else:
        custom_legend = [Patch(edgecolor='grey', label="RF", facecolor='green'),
                         Line2D([0], [0], color='red', lw=1.5, label="Pre-CD1"),
                         Patch(edgecolor='grey', label="XGBoost", facecolor='orange'),
                         Line2D([0], [0], color='black', lw=1.5, label="Post-CD1"),
                         Patch(edgecolor='grey', label="LSTM", facecolor='blue'),
                         Line2D([0], [0], color='grey', lw=1.5, label="Failured")]

    plt.xticks([r + bar_width for r in range(len(model1))],
               ['', '', '', '', '', '', '', '', '', '', '', ''])

    plt.grid(axis='y', ls="--", lw=0.5, zorder=1)
    plt.ylabel('Increased Warning Time\n(minute)', fontweight='bold')
    plt.ylim(0, 240)
    plt.xlim(-0.25, 11.75)
    plt.text(x=min(x1) - bar_width, y=210, s=f' {idx}', weight="bold")

def benckmark_result(y, x_start, x_end, warning_type, c):
    if warning_type == "pre-cd1":
        plt.hlines(y=y, xmin=x_start, xmax=x_end, color=c, linewidth=1.5, label="Pre-CD1 Warning")
    elif warning_type == "post-cd1":
        plt.hlines(y=y, xmin=x_start, xmax=x_end, color=c, linewidth=1.5, label="Post-CD1 Warning")



fig = plt.figure(figsize=(5, 4))

gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])


# <editor-fold desc="velocity volume">
ax = plt.subplot(gs[0])
df = pd.read_csv("/Users/qizhou/#file/2_projects/FI_paper/1figs/1studyArea/volume_velocity.txt", skiprows=22, header=None)
volume = np.array(df.iloc[:, 2]) # m3
velocity = np.array(df.iloc[:, 3]) # m/s

volume[volume=='No Data'] = np.nan
velocity[velocity=='No Data'] = np.nan

volume = volume.astype(float)/1e4
velocity = velocity.astype(float)


bar_width = 0.35
x = np.arange(volume.size)
x1 = [x - bar_width/2 for x in x]
x2 = [x + bar_width/2 for x in x]

for step in range(volume.size):
    if np.isnan(volume[step]):
        ax.bar(x1[step], 100, width=bar_width, edgecolor='grey', label="Volume", color='grey', alpha=0.3)
    else:
        ax.bar(x1[step], volume[step], width=bar_width, edgecolor='grey', label="Volume", color='green')

plt.yticks([0, 2.5, 5, 7.5, 10], ["0", "2.5", "5.0", "7.5", "10"])
plt.ylabel("Volume\n"+r"($\times10^4 "+ "m" + r"^3$)", fontweight='bold')

plt.text(x=min(x1)-bar_width, y=11, s=' (a)', weight="bold")
plt.xlim(min(x1)-bar_width, max(x2)+bar_width)
plt.ylim(0, 13)

ax1 = ax.twinx()
for step in range(velocity.size):
    if np.isnan(velocity[step]):
        ax1.bar(x2[step], 100, width=bar_width, edgecolor='grey', label="Volume", color='grey', alpha=0.3)
    else:
        ax1.bar(x2[step], velocity[step], width=bar_width, edgecolor='grey', label="Volume", color='orange')
plt.ylim(0, 4)

plt.yticks([0, 1, 2, 3, 4], ["0", "2", "2", "3", "4"])
plt.ylabel("Velocity (m/s)", fontweight='bold')

ax.axes.xaxis.set_ticklabels([])
#plt.xticks(x, ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7a', 'E7b', 'E8', 'E9', 'E10', 'E11'])

custom_legend = [Patch(edgecolor='grey', label="Volume", facecolor='green'),
                 Patch(edgecolor='grey', label="Velocity", facecolor='orange'),
                 Patch(edgecolor='grey', label="No Data", facecolor='grey', alpha=0.3)]

ax1.legend(handles=custom_legend, loc="upper right", fontsize=6)
plt.xticks([r for r in range(velocity.size)],
           ['', '', '', '', '', '', '', '', '', '', '', ''])
# </editor-fold>



ax = plt.subplot(gs[1])
feature_type = "A"
pre_cd1, post_cd1 = gosia_warning()
warning1 = fetch_data("Random_Forest", feature_type, "", 0, 0.2, 10) # no false warning and missed
warning2 = fetch_data("XGBoost", feature_type, "", 0, 0.2, 10) # no false warning and missed
warning3 = fetch_data("LSTM", feature_type, "", 0, 0.2, 10)
print(np.nanmean(np.concatenate((warning1[:4], warning1[6:])))/60, np.nansum(np.concatenate((warning1[:4], warning1[6:])))/60,
      np.nanmean(np.concatenate((warning2[:4], warning2[6:])))/60, np.nansum(np.concatenate((warning2[:4], warning2[6:])))/60,
      np.nanmean(np.concatenate((warning3[:4], warning3[6:])))/60, np.nansum(np.concatenate((warning3[:4], warning3[6:])))/60)

plot_bar(ax, warning1/60, warning2/60, warning3/60, pre_cd1, post_cd1, "(b) Type A Feature", True)


ax = plt.subplot(gs[2])
feature_type = "C"
pre_cd1, post_cd1 = gosia_warning()
warning1 = fetch_data("Random_Forest", feature_type, "", 0, 0.4, 2) # no false warning and missed
warning2 = fetch_data("XGBoost", feature_type, "", 0, 0.4, 2) # no false warning and missed
warning3 = fetch_data("LSTM", feature_type, "", 0, 0.4, 2)
print(np.nanmean(np.concatenate((warning1[:4], warning1[6:])))/60, np.nansum(np.concatenate((warning1[:4], warning1[6:])))/60,
      np.nanmean(np.concatenate((warning2[:4], warning2[6:])))/60, np.nansum(np.concatenate((warning2[:4], warning2[6:])))/60,
      np.nanmean(np.concatenate((warning3[:4], warning3[6:])))/60, np.nansum(np.concatenate((warning3[:4], warning3[6:])))/60)

plot_bar(ax, warning1/60, warning2/60, warning3/60, pre_cd1, post_cd1, "(c) Type C Feature")

plt.xticks(np.arange(12)+0.25, ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7a', 'E7b', 'E8', 'E9', 'E10', 'E11'])
plt.xlabel('Event Index', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{parent_dir}/warning_time_difference.png", dpi=600)
plt.show()


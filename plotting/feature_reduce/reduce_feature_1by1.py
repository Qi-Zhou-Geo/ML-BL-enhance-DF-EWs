#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')=
import os
from datetime import datetime
import seaborn as sns
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.ticker as ticker

# </editor-fold>




plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )#, 'font.weight':'bold'
parent_dir = os.path.dirname(os.path.abspath(__file__))  # get the parent path



def feath_data(station, model_type):
    df = pd.read_csv(f"{parent_dir}/reduce_1by1.txt", header=None)
    arr = np.array(df)

    id1 = np.where(arr[:, 0] == station)[0]
    id2 = np.where(arr[:, 1] == model_type)[0]

    arrays = [id1, id2]
    intersection = arrays[0]
    for arr_step in arrays[1:]:
        intersection = np.intersect1d(intersection, arr_step)

    intersection_arr = arr[intersection, :]
    intersection_arr = intersection_arr[intersection_arr[:, 3].argsort()[::-1]]

    tn, fp, fn, tp = intersection_arr[:, 7], intersection_arr[:, 9], intersection_arr[:, 11], intersection_arr[:, 13]
    f1 = intersection_arr[:, 15]
    falilured_detected = intersection_arr[:, 19]

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    return f1, falilured_detected, tpr, fpr


def plot_fig(ax, tpr, fpr, model_type):

    if model_type == "Random_Forest":
        c = 'b'
        model_type = "Random Forest"
    elif model_type == "XGBoost":
        c = 'r'
        model_type = "XGBoost"
    else:
        print(f"check model type {model_type}")

    x = np.arange(tpr.size)

    plt.scatter(x, tpr, marker="o", color=c, alpha=0.5, label=f"TPR of {model_type}", zorder=2)
    plt.scatter(x, fpr, marker="s", color=c, alpha=0.5, label=f"FPR of {model_type}", zorder=2)

    plt.grid(axis='y', ls="--", lw=0.5, zorder=1)

    plt.yscale("log")
    plt.ylim(1e-4, 2)
    plt.xlim(-0.5, 80.5)
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2)) # unit is mintute



fig = plt.figure(figsize=(5.5, 4))
gs = gridspec.GridSpec(3, 1)

ax = plt.subplot(gs[0])
station, model_type = "ILL18", "Random_Forest"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plot_fig(ax, tpr, fpr, model_type)

station, model_type = "ILL18", "XGBoost"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plot_fig(ax, tpr, fpr, model_type)
plt.legend(loc="center right", fontsize=6, ncols=2)
plt.text(x=0, y=1e-1, s=f" (a) {station}", weight="bold")


ax = plt.subplot(gs[1])
station, model_type = "ILL12", "Random_Forest"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plot_fig(ax, tpr, fpr, model_type)

station, model_type = "ILL12", "XGBoost"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plot_fig(ax, tpr, fpr, model_type)
plt.text(x=0, y=1e-1, s=f" (b) {station}", weight="bold")


ax = plt.subplot(gs[2])
station, model_type = "ILL13", "Random_Forest"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plot_fig(ax, tpr, fpr, model_type)

station, model_type = "ILL13", "XGBoost"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plot_fig(ax, tpr, fpr, model_type)

plt.text(x=0, y=1e-1, s=f" (c) {station}", weight="bold")

xLocation = np.arange(0, 81, 10)
xTicker = [str(80-i) for i in xLocation]
ax.set_xticks(xLocation, xTicker)
plt.xlabel("Number of Input Seismic Features", weight="bold")


fig.text(x=0, y=0.5, s="Ture Positive Rate (TPR) or False Positive Rate (FPR)", weight="bold", va='center', rotation='vertical')

plt.tight_layout()
plt.savefig(f"{parent_dir}/reduce_feature_1by1.png", dpi=600, transparent=True)
plt.show()

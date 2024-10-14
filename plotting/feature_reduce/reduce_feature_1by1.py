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


def plot_fig(ax, tpr, fpr, model_type, idx):

    if model_type == "Random_Forest":
        marker = 'o'
        model_type = "Random Forest"
        ls = "--"
    elif model_type == "XGBoost":
        marker = 's'
        model_type = "XGBoost"
        ls = "-"
    else:
        print(f"check model type {model_type}")

    x = np.arange(tpr.size)

    #ax.scatter(x, tpr, marker=marker, color="black", alpha=0.5, label=f"TPR of {model_type}", zorder=2)
    ax.plot(x, tpr, color="black", ls=ls, label=f"{model_type}", zorder=2)
    ax.grid(axis='x', ls="--", lw=0.5, zorder=1)
    ax.spines['left'].set_color("black")
    ax.tick_params(axis='y', which='both', colors="black")
    if idx == 1:
        plt.ylabel("True Positive Rate (TPR)", weight="bold", color="black")


    ax.set_ylim(0.2, 1.1)
    plt.xlim(-0.5, 80.5)
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2)) # unit is mintute

    ax0 = ax.twinx()
    #ax0.scatter(x, fpr, marker=marker, color="red", alpha=0.5, label=f"FPR of {model_type}", zorder=2)
    ax0.plot(x, fpr, color="red", ls=ls, label=f"{model_type}", zorder=2)
    ax0.spines['right'].set_color("red")
    ax0.tick_params(axis='y', which='both', colors="red")
    if idx == 1:
        plt.ylabel("False Positive Rate (FPR)", weight="bold", color="red")


    plt.yscale("log")
    ax0.set_ylim(1e-4, 1e-2)
    plt.xlim(-0.5, 80.5)
    ax0.axes.xaxis.set_ticklabels([])
    ax0.xaxis.set_minor_locator(ticker.MultipleLocator(2)) # unit is mintute

    return ax, ax0




fig = plt.figure(figsize=(5.5, 4))
gs = gridspec.GridSpec(3, 1)

ax = plt.subplot(gs[0])
station, model_type = "ILL18", "Random_Forest"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plt.text(x=0, y=1.0, s=f" (a) {station}", weight="bold")
plot_fig(ax, tpr, fpr, model_type, 0)

station, model_type = "ILL18", "XGBoost"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
ax, ax0 = plot_fig(ax, tpr, fpr, model_type, 0)
#plt.legend(loc="center right", fontsize=6, ncols=2)


ax = plt.subplot(gs[1])
station, model_type = "ILL12", "Random_Forest"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plt.text(x=0, y=1.0, s=f" (b) {station}", weight="bold")
plot_fig(ax, tpr, fpr, model_type, 1)

station, model_type = "ILL12", "XGBoost"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plot_fig(ax, tpr, fpr, model_type, 0)
from matplotlib.lines import Line2D
# Create the custom legend with different line styles and colors
handles = [
    Line2D([], [], color="black", linestyle='--', label="TPR of Random Forest"),
    Line2D([], [], color="red", linestyle='--', label="FPR of Random Forest"),
    Line2D([], [], color="black", linestyle='-', label="TPR of XGBoost"),   # Dashed black line
    Line2D([], [], color="red", linestyle='-', label="FPR of XGBoost"),     # Dashed red line
]
# Add the custom legend to the plot
labels = ["TPR of Random Forest", "FPR of Random Forest", "TPR of XGBoost", "FPR of XGBoost"]
plt.legend(handles, labels, loc="lower left", ncol=2)


ax = plt.subplot(gs[2])
station, model_type = "ILL13", "Random_Forest"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plt.text(x=0, y=1.0, s=f" (c) {station}", weight="bold")
plot_fig(ax, tpr, fpr, model_type, 2)

station, model_type = "ILL13", "XGBoost"
f1, falilured_detected, tpr, fpr = feath_data(station, model_type)
plot_fig(ax, tpr, fpr, model_type, 2)

xLocation = np.arange(0, 81, 10)
xTicker = [str(80-i) for i in xLocation]
ax.set_xticks(xLocation, xTicker)
ax.set_xlabel("Number of Input Seismic Features", weight="bold")

plt.tight_layout()
plt.savefig(f"{parent_dir}/reduce_feature_1by1.png", dpi=600)
plt.show()

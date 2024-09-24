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

def plot_vertical_line(data_start, cd29, sps):

    label = UTCDateTime(cd29) - UTCDateTime(data_start)
    plt.axvline(x=label * sps, color="red", lw=1, ls="--", zorder=1)



# meta information, time is UTC+0
seismic_network, input_component = "9S", "EHZ"
cd29 = "2022-06-05  11:33:00"
data_start, data_end = "2022-06-05 09:00:00", "2022-06-05 13:00:00"
x_interval = 1 # unit is hour


st1 = load_seismic_signal(seismic_network, "ILL18", input_component, data_start, data_end)
st2 = load_seismic_signal(seismic_network, "ILL12", input_component, data_start, data_end)
st3 = load_seismic_signal(seismic_network, "ILL13", input_component, data_start, data_end)


fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1])



ax = plt.subplot(gs[0])
st1.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
ax.images[0].set_clim(-180, -100)
plt.ylabel("Frequency 18(Hz)", fontweight='bold')
plt.yticks([1, 10, 20, 30, 40, 45], [1, 10, 20, 30, 40, 45])
plt.ylim(1, 45)
plot_vertical_line(data_start, cd29, st1[0].stats.sampling_rate)
ax.axes.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval))  # unit is second
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="7%", pad=0.05)
cbar = plt.colorbar(mappable=ax.images[-1], cax=cax, orientation='horizontal')
cbar.ax.xaxis.set_ticks_position('top')  # Move color bar tick labels to the top
cbar.set_label('Power Spectral Density PSD (dB)', fontsize=6, labelpad=-25, fontweight="bold")


ax = plt.subplot(gs[1])
st2.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
ax.images[0].set_clim(-180, -100)
plt.ylabel("Frequency 12(Hz)", fontweight='bold')
plt.yticks([1, 10, 20, 30, 40, 45], [1, 10, 20, 30, 40, 45])
plt.ylim(1, 45)
plot_vertical_line(data_start, cd29, st1[0].stats.sampling_rate)
ax.axes.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval))  # unit is second


ax = plt.subplot(gs[2])
st3.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
ax.images[0].set_clim(-180, -100)
plt.ylabel("Frequency 13(Hz)", fontweight='bold')
plt.yticks([1, 10, 20, 30, 40, 45], [1, 10, 20, 30, 40, 45])
plt.ylim(1, 45)
plot_vertical_line(data_start, cd29, st1[0].stats.sampling_rate)
ax.axes.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval))  # unit is second




ax = plt.subplot(gs[3])
ax.plot(st1[0].data, lw=1, label=f"{seismic_network}-ILL18-{input_component}", color="black")
ax.plot(st2[0].data, lw=1, label=f"{seismic_network}-ILL12-{input_component}", color="black")
ax.plot(st3[0].data, lw=1, label=f"{seismic_network}-ILL13-{input_component}", color="black")
plot_vertical_line(data_start, cd29, st1[0].stats.sampling_rate)
plt.xlim(0, st1[0].data.size-1)
ax.axes.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval * st1[0].stats.sampling_rate))  # unit is second
plt.ylabel("Amplitude (m/s)", fontweight='bold')
plt.legend(loc="upper right", fontsize=6)
plt.grid(axis='y', ls="--", lw=0.5, zorder=1)
plt.xlabel(f"UTC+0 (minute-by-minute since {data_start})", fontweight='bold')


plt.tight_layout()
plt.savefig(f"{CONFIG_dir['parent_dir']}/plotting/dual_testing_pro/"
            f"{seismic_network}_2022-06-05_{input_component}_{data_start}_{data_end}.png", dpi=600)

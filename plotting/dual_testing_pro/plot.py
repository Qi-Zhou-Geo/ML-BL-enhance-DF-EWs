#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
import os
import sys
import argparse

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


def plot(seismic_network, input_component, x_interval, data_start, data_end, julday, step):
    st1 = load_seismic_signal(seismic_network, "ILL17", input_component, data_start, data_end)
    st2 = load_seismic_signal(seismic_network, "ILL12", input_component, data_start, data_end)
    st3 = load_seismic_signal(seismic_network, "ILL13", input_component, data_start, data_end)

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1])

    ax = plt.subplot(gs[0])
    st1.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
    ax.images[0].set_clim(-180, -100)
    plt.ylabel("Frequency (Hz)", fontweight='bold')
    plt.yticks([1, 10, 20, 30, 40, 45], [1, 10, 20, 30, 40, 45])
    plt.ylim(1, 45)
    plt.text(x=0, y=35, color="white", s=" ILL17")
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
    plt.ylabel("Frequency (Hz)", fontweight='bold')
    plt.yticks([1, 10, 20, 30, 40, 45], [1, 10, 20, 30, 40, 45])
    plt.ylim(1, 45)
    plt.text(x=0, y=35, color="white", s=" ILL12")
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval))  # unit is second

    ax = plt.subplot(gs[2])
    st3.spectrogram(per_lap=0.5, wlen=60, log=False, dbscale=True, mult=True, title="", axes=ax, cmap='inferno')
    ax.images[0].set_clim(-180, -100)
    plt.ylabel("Frequency (Hz)", fontweight='bold')
    plt.yticks([1, 10, 20, 30, 40, 45], [1, 10, 20, 30, 40, 45])
    plt.ylim(1, 45)
    plt.text(x=0, y=35, color="white", s=" ILL13")
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval))  # unit is second

    ax = plt.subplot(gs[3])
    ax.plot(st1[0].data, lw=1, label=f"{seismic_network}-ILL17-{input_component}")
    ax.plot(st2[0].data, lw=1, label=f"{seismic_network}-ILL12-{input_component}")
    ax.plot(st3[0].data, lw=1, label=f"{seismic_network}-ILL13-{input_component}")
    plt.xlim(0, st1[0].data.size - 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval * st1[0].stats.sampling_rate))  # unit is second
    plt.ylabel("Amplitude (m/s)", fontweight='bold')
    plt.legend(loc="upper right", fontsize=6)
    plt.grid(axis='y', ls="--", lw=0.5, zorder=1)
    plt.xlabel(f"UTC+0 (minute-by-minute since {data_start})", fontweight='bold')

    duration = int((UTCDateTime(data_end) - UTCDateTime(data_start)) / 3600)
    xLocation = np.arange(0, 60 * (duration + x_interval), 60 * x_interval)
    xTicks = [(UTCDateTime(data_start) + i * 60).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
    plt.xticks(xLocation * 60 * st1[0].stats.sampling_rate, xTicks)

    plt.tight_layout()
    plt.savefig(f"{CONFIG_dir['parent_dir']}/plotting/dual_testing_pro/2022/"
                f"{julday}_{str(step).zfill(3)}_{seismic_network}_{input_component}_{data_start}_{data_end}.png", dpi=600)


def maaa1(julday):

    for step in np.arange(0, 6):
        data_start = UTCDateTime(year=2022, julday=julday) + step * 4 * 3600
        data_end = UTCDateTime(year=2022, julday=julday) + (step + 1) * 4 * 3600

        data_start, data_end = data_start.strftime('%Y-%m-%d %H:%M:%S'), data_end.strftime('%Y-%m-%d %H:%M:%S')

        seismic_network, input_component, x_interval = "9S", "EHZ", 1
        plot(seismic_network, input_component, x_interval, data_start, data_end, julday, step)

def main(julday):
    step = 1
    data_start = UTCDateTime(year=2022, julday=julday)
    data_end = UTCDateTime(year=2022, julday=julday) + 24 * 3600

    data_start, data_end = data_start.strftime('%Y-%m-%d %H:%M:%S'), data_end.strftime('%Y-%m-%d %H:%M:%S')

    seismic_network, input_component, x_interval = "9S", "EHZ", 4
    plot(seismic_network, input_component, x_interval, data_start, data_end, julday, step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--julday", type=float)

    args = parser.parse_args()

    main(args.julday)


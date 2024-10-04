#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
import os
import pytz
import numpy as np
import pandas as pd
from datetime import datetime
from obspy import read, Stream, UTCDateTime, read_inventory
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


import matplotlib
matplotlib.use('TkAgg')  # Or 'TkAgg'
import matplotlib.pyplot as plt

plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # get the parent path


def get_seismicSignal(data_name, data_start, data_end):
    inv = read_inventory("/Users/qizhou/#SAC/2017-2020/metadata_2017-2020.xml")

    st = read(f"/Users/qizhou/#SAC/2017-2020/{data_name}")
    d1 = UTCDateTime(data_start)
    d2 = UTCDateTime(data_end)
    st = st.trim(starttime=d1, endtime=d2, nearest_sample=False)
    st.merge(method=1, fill_value='interpolate')
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    st.filter("bandpass", freqmin=1, freqmax=45)
    st.remove_response(inventory=inv)
    st.detrend('linear')
    st.detrend('demean')

    return st


def fetc_data(input_station, model_type, feature_type, input_component, data_start, data_end):

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # get the parent path
    df = pd.read_csv(f"{parent_dir}/output/train_test_output/predicted_results/"
                     f"{input_station}_{model_type}_{feature_type}_{input_component}_testing_output.txt")
    date = np.array(df.iloc[:, 0])

    id1 = np.where(date == data_start)[0][0]
    id2 = np.where(date == data_end)[0][0] + 1

    pro = np.array(df.iloc[id1:id2, 3])

    return pro


def plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval):

    plt.plot(pro18, label=f"ILL18", linewidth=1, color="blue", zorder=2)#3B75AF")
    plt.plot(pro12, label=f"ILL12", linewidth=1, color="orange", zorder=2)#EF8636")
    plt.plot(pro13, label=f"ILL13", linewidth=1, color="green", zorder=2)#"#519E3E")

    plt.plot(final_warning, label=f"Final Warning", linewidth=2, color="black", zorder=3)#"#519E3E")

    plt.xlim(0, len(pro18))

    plt.grid(axis='y', ls="--", lw=0.5, zorder=1)

    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60 * x_interval)) # unit is mintute
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3), useMathText=True)

    plt.ylabel(f"Debris flow\nProbability", fontweight="bold")
    yLocation = np.arange(0, 1.1, 0.25)
    yTicks = [str(round(label,2)) for label in yLocation]
    plt.yticks(yLocation, ["0", "0.25", "0.50", "0.75", "1.0"])


def plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, sps):

    label1 = UTCDateTime(pre_cd1_warning) - UTCDateTime(data_start)
    label2 = UTCDateTime(post_cd1_warning) - UTCDateTime(data_start)
    label3 = UTCDateTime(cd29) - UTCDateTime(data_start)

    plt.axvline(x=label1 * sps, color="red", lw=1, ls="--", zorder=1)
    plt.axvline(x=label2 * sps, color="red", lw=1, ls="-", zorder=1)
    plt.axvline(x=label3 * sps, color="black", lw=1, ls="-", zorder=1)


def warning_pro(model_type, feature_type, input_component, data_start, data_end, pro_threshold, warning_threshold, attention_window_size):

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # get the parent path
    df = pd.read_csv(f"{parent_dir}/plotting/network_warning/output/"
                     f"{model_type}_{feature_type}_{input_component}_warning_{pro_threshold}_{warning_threshold}_{attention_window_size}.txt",
                     header=0)
    date = np.array(df.iloc[:, 0])

    id1 = np.where(date == data_start)[0][0]
    id2 = np.where(date == data_end)[0][0] + 1

    pro = np.array(df.iloc[id1:id2, -2])
    pro[np.where(pro == 'noise')] = 0
    pro[np.where(pro == 'false_warning')] = 0.5
    pro[np.where(pro == 'fake_warning')] = 0.5
    pro = pro.astype(float) / 60

    pro[np.where(pro > 0)] = 1
    return pro


x_interval = 1
data_start, data_end = "2020-06-29 03:30:00", "2020-06-29 07:30:00"
pre_cd1_warning, post_cd1_warning =  "2020-06-29 04:33:29", "2020-06-29 05:10:09"
cd29 = "2020-06-29 05:49:13"
pro_threshold, warning_threshold, attention_window_size = 0, 0.4, 2
input_component, feature_type = "EHZ", 'C'


# <editor-fold desc="plot">
fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(4, 1)

ax = plt.subplot(gs[0])
order = [3, 1, 2]
c = ["blue", "orange", "green"]
for idx, station in enumerate(["ILL18", "ILL12", "ILL13"]):
    julday = UTCDateTime(data_start).julday
    st = get_seismicSignal(f"9S.{station}.EHZ.2020.{julday}", data_start, data_end)
    ax.plot(st[0].data, lw=1.5, label=station, zorder=order[idx], color=c[idx])

plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 100)

ax.set_title(r"x$10^{-5}$", loc="left", fontsize=6)
ax.axes.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval * 100))  # unit is saecond
ax.yaxis.set_major_locator(ticker.MultipleLocator(1e-4/2))  # unit is saecond
plt.grid(axis='y', ls="--", lw=0.5, zorder=1)
plt.ylabel("Amplitude (m/s)", fontweight='bold')
plt.ylim(-1e-4, 1e-4)
plt.xlim(0, st[0].data.size-1)
plt.yticks([-1e-4, 5*-1e-5, 0, 5*1e-5, 1e-4], ["-10", "-5", "0", "5", "10"])
plt.text(x=0, y=1e-4*0.8, s=" (a)", fontweight='bold')



ax = plt.subplot(gs[1])
model_type = "Random_Forest"
pro18 = fetc_data("ILL18", model_type, feature_type, input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, feature_type, input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, feature_type, input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
plt.text(x=0, y=0.8, s=f" (b) Random Forest", fontweight='bold')
plt.legend(loc="upper right", fontsize=6)


ax = plt.subplot(gs[2])
model_type = "XGBoost"
pro18 = fetc_data("ILL18", model_type, feature_type, input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, feature_type, input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, feature_type, input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
plt.text(x=0, y=0.8, s=f" (c) {model_type}", fontweight='bold')


ax = plt.subplot(gs[3])
model_type = "LSTM"
pro18 = fetc_data("ILL18", model_type, feature_type, input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, feature_type, input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, feature_type, input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
plt.text(x=0, y=0.8, s=f" (d) {model_type}", fontweight='bold')

duration = int((UTCDateTime(data_end) - UTCDateTime(data_start))/3600)
xLocation = np.arange(0, 60 * (duration + x_interval), 60 * x_interval)
xTicks = [(UTCDateTime(data_start) + i * 60).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
plt.xticks(xLocation, xTicks)
plt.xlabel(f"UTC+0 Time", fontweight='bold')



plt.tight_layout()
plt.savefig(f"{parent_dir}/plotting/amp_pro/amp_pro_{data_end}_{data_end}.png", dpi=600)
plt.show()
# </editor-fold>



x_interval = 2
data_start, data_end = "2020-06-17 00:00:00", "2020-06-17 08:00:00"
pre_cd1_warning, post_cd1_warning = "2020-06-17 03:11:48", "2020-06-17 03:40:08"
cd29 = "2020-06-17 04:06:58"
pro_threshold, warning_threshold, attention_window_size = 0, 0.4, 2
input_component, feature_type = "EHZ", 'C'


# <editor-fold desc="plot">
fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(4, 1)

ax = plt.subplot(gs[0])
order = [3, 1, 2]
c = ["blue", "orange", "green"]
for idx, station in enumerate(["ILL18", "ILL12", "ILL13"]):
    julday = UTCDateTime(data_start).julday
    st = get_seismicSignal(f"9S.{station}.EHZ.2020.{julday}", data_start, data_end)
    ax.plot(st[0].data, lw=1.5, label=station, zorder=order[idx], color=c[idx])


plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 100)
#"2020-06-17 09:00:00" this is not exist and will not display in the figure
plot_vertical_line(data_start, "2020-06-17 09:00:00", "2020-06-17 01:51:48", cd29, 100)

t1 = ["2020-06-17 01:10:00","2020-06-17 01:50:00","2020-06-17 01:30:00","2020-06-17 02:45:00","2020-06-17 01:50:00","2020-06-17 02:00:00"]
t2 = ["2020-06-17 03:05:00","2020-06-17 04:30:00","2020-06-17 03:25:00","2020-06-17 05:20:00","2020-06-17 03:40:00","2020-06-17 06:05:00"]
c = ["blue", "blue", "orange", "orange", "green", "green"]

for idx, item1 in enumerate(t1):
    sps = 100
    label1 = UTCDateTime(item1) - UTCDateTime(data_start)

    if idx % 2 == 0:
        style = "--"
    else:
        style = "-"
    plt.axvline(x=label1 * sps, color=c[idx], lw=1, ls=style, zorder=1)

for idx, item1 in enumerate(t2):
    sps = 100
    label1 = UTCDateTime(item1) - UTCDateTime(data_start)

    if idx % 2 == 0:
        style = "--"
    else:
        style = "-"
    plt.axvline(x=label1 * sps, color=c[idx], lw=2, ls=style, zorder=1)



ax.set_title(r"x$10^{-5}$", loc="left", fontsize=6)
ax.axes.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval * 100))  # unit is saecond
ax.yaxis.set_major_locator(ticker.MultipleLocator(1e-4/2))  # unit is saecond
plt.grid(axis='y', ls="--", lw=0.5, zorder=1)
plt.ylabel("Amplitude (m/s)", fontweight='bold')
plt.ylim(-1e-4, 1e-4)
plt.xlim(0, st[0].data.size-1)
plt.yticks([-1e-4, 5*-1e-5, 0, 5*1e-5, 1e-4], ["-10", "-5", "0", "5", "10"])
plt.text(x=0, y=1e-4*0.8, s=" (a)", fontweight='bold')



ax = plt.subplot(gs[1])
model_type = "Random_Forest"
pro18 = fetc_data("ILL18", model_type, feature_type, input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, feature_type, input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, feature_type, input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
#"2020-06-17 09:00:00" this is not exist and will not display in the figure
plot_vertical_line(data_start, "2020-06-17 09:00:00", "2020-06-17 01:51:48", cd29, 1/60)

plt.text(x=0, y=0.8, s=f" (b) Random Forest", fontweight='bold')
plt.legend(loc="upper right", fontsize=6)


ax = plt.subplot(gs[2])
model_type = "XGBoost"
pro18 = fetc_data("ILL18", model_type, feature_type, input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, feature_type, input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, feature_type, input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
#"2020-06-17 09:00:00" this is not exist and will not display in the figure
plot_vertical_line(data_start, "2020-06-17 09:00:00", "2020-06-17 01:51:48", cd29, 1/60)
plt.text(x=0, y=0.8, s=f" (c) {model_type}", fontweight='bold')


ax = plt.subplot(gs[3])
model_type = "LSTM"
pro18 = fetc_data("ILL18", model_type, feature_type, input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, feature_type, input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, feature_type, input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
#"2020-06-17 09:00:00" this is not exist and will not display in the figure
plot_vertical_line(data_start, "2020-06-17 09:00:00", "2020-06-17 01:51:48", cd29, 1/60)
plt.text(x=0, y=0.8, s=f" (d) {model_type}", fontweight='bold')

duration = int((UTCDateTime(data_end) - UTCDateTime(data_start))/3600)
xLocation = np.arange(0, 60 * (duration + x_interval), 60 * x_interval)
xTicks = [(UTCDateTime(data_start) + i * 60).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
plt.xticks(xLocation, xTicks)
plt.xlabel(f"UTC+0 Time", fontweight='bold')



plt.tight_layout()
plt.savefig(f"{parent_dir}/plotting/amp_pro/amp_pro_{data_end}_{data_end}.png", dpi=600)
plt.show()
# </editor-fold>




x_interval = 1
data_start, data_end = "2020-06-29 03:30:00", "2020-06-29 07:30:00"
pre_cd1_warning, post_cd1_warning =  "2020-06-29 04:33:29", "2020-06-29 05:10:09"
cd29 = "2020-06-29 05:49:13"
pro_threshold, warning_threshold, attention_window_size = 0, 0.4, 2
input_component, feature_type = "EHZ", 'C'




# <editor-fold desc="SI1">
fig = plt.figure(figsize=(5, 4))
gs = gridspec.GridSpec(4, 1)

ax = plt.subplot(gs[0])
order = [3, 1, 2]
c = ["blue", "orange", "green"]
for idx, station in enumerate(["ILL18", "ILL12", "ILL13"]):
    julday = UTCDateTime(data_start).julday
    st = get_seismicSignal(f"9S.{station}.EHZ.2020.{julday}", data_start, data_end)
    ax.plot(st[0].data, lw=1.5, label=station, zorder=order[idx], color=c[idx])


plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 100)

ax.set_title(r"x$10^{-5}$", loc="left", fontsize=6)
ax.axes.xaxis.set_ticklabels([])
ax.xaxis.set_major_locator(ticker.MultipleLocator(3600 * x_interval * 100))  # unit is saecond
ax.yaxis.set_major_locator(ticker.MultipleLocator(1e-4/2))  # unit is saecond
plt.grid(axis='y', ls="--", lw=0.5, zorder=1)
plt.ylabel("Amplitude (m/s)", fontweight='bold')
plt.ylim(-1e-4, 1e-4)
plt.xlim(0, st[0].data.size-1)
plt.yticks([-1e-4, 5*-1e-5, 0, 5*1e-5, 1e-4], ["-10", "-5", "0", "5", "10"])
plt.text(x=0, y=1e-4*0.8, s=" (a)", fontweight='bold')



ax = plt.subplot(gs[1])
model_type = "Random_Forest"
pro18 = fetc_data("ILL18", model_type, "A", input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, "B", input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, "C", input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
plt.text(x=0, y=0.8, s=f" (b) Random Forest", fontweight='bold')
plt.legend(loc="upper right", fontsize=6)


ax = plt.subplot(gs[2])
model_type = "XGBoost"
pro18 = fetc_data("ILL18", model_type, feature_type, input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, feature_type, input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, feature_type, input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
plt.text(x=0, y=0.8, s=f" (c) {model_type}", fontweight='bold')


ax = plt.subplot(gs[3])
model_type = "LSTM"
pro18 = fetc_data("ILL18", model_type, feature_type, input_component, data_start, data_end)
pro12 = fetc_data("ILL12", model_type, feature_type, input_component, data_start, data_end)
pro13 = fetc_data("ILL13", model_type, feature_type, input_component, data_start, data_end)
final_warning = warning_pro(model_type, feature_type, input_component, data_start, data_end,
                            pro_threshold, warning_threshold, attention_window_size)
plot_model_pro(pro18, pro12, pro13, final_warning, ax, x_interval)
plot_vertical_line(data_start, pre_cd1_warning, post_cd1_warning, cd29, 1/60)
plt.text(x=0, y=0.8, s=f" (d) {model_type}", fontweight='bold')

duration = int((UTCDateTime(data_end) - UTCDateTime(data_start))/3600)
xLocation = np.arange(0, 60 * (duration + x_interval), 60 * x_interval)
xTicks = [(UTCDateTime(data_start) + i * 60).strftime('%Y-%m-%d' + '\n' + '%H:%M:%S') for i in xLocation]
plt.xticks(xLocation, xTicks)
plt.xlabel(f"UTC+0 Time", fontweight='bold')



plt.tight_layout()
plt.savefig(f"{parent_dir}/plotting/amp_pro/amp_pro_{data_end}_{data_end}.png", dpi=600)
plt.show()
# </editor-fold>

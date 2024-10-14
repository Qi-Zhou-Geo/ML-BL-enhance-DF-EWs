#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
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

# Specify the directory containing the Arial font
from matplotlib import font_manager
font_dirs = ['/storage/vast-gfz-hpc-01/home/qizhou/2python/font']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

# Add fonts to the FontManager
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)



plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )#, 'font.weight':'bold'
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
input_component = "EHZ"



scaler = MinMaxScaler(feature_range=(0, 1))

def visualizeFeatureIMP(y1, y2):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    y1 = np.array(y1).reshape(-1, 1)
    y2 = np.array(y2).reshape(-1, 1)

    y1 = scaler.fit_transform(y1).flatten()
    y2 = scaler.fit_transform(y2).flatten()


    x = np.arange(y1.size)

    markerline1, stemlines1, baseline1 = plt.stem(x, y1, linefmt='black', markerfmt="ko", label="Random Forest")
    markerline1.set_markersize(6)
    markerline1.set_alpha(0.5)
    baseline1.set_visible(False)

    markerline2, stemlines2, baseline2 = plt.stem(x, y2, linefmt='red', markerfmt="ro", label="XGBoost")
    markerline2.set_markersize(6)
    markerline2.set_alpha(0.5)
    baseline2.set_visible(False)

    featureIDboundary = np.array([[-0.5, 10], [11, 35], [36, 52], [53, 69], [70, 79.5]])
    for step in range(featureIDboundary.shape[0]):
        if step % 2 == 0:
            facecolor = "black"
        else:
            facecolor = "grey"
        plt.axvspan(xmin=featureIDboundary[step, 0], xmax=featureIDboundary[step, 1],
                    ymin=0, ymax=1, alpha=0.2, edgecolor="None", facecolor=facecolor)

    #for step in [0, 8, 10, 23, 24, 35]:
        #plt.axvline(x=step, color="green", lw=0.8, ls="--", zorder=0)

    plt.xlim(-0.5, 79.5)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', ls="--", lw=0.5)
    plt.yticks([0, 0.25, 0.50, 0.75, 1.0], ["0.0", "0.25", "0.50", "0.75", "1.0"])


y1_sum = np.full(80, 0)
y2_sum = np.full(80, 0)

fig = plt.figure(figsize=(5.5, 4))
gs = gridspec.GridSpec(3, 1)

# <editor-fold desc="ILL18">
ax1 = plt.subplot(gs[0])
#ax1.set_title('(a) ILL18', weight="bold", loc='left')
input_station, feature_type = "ILL18", "C"
model_type = "Random_Forest"
df1 = pd.read_csv(f"{parent_dir}/output/train_test_output/figures/"
                  f"{input_station}_{model_type}_{feature_type}_{input_component}_build_in_IMP.txt", header = None)
model_type = "XGBoost"
df2 = pd.read_csv(f"{parent_dir}/output/train_test_output/figures/"
                  f"{input_station}_{model_type}_{feature_type}_{input_component}_build_in_IMP.txt", header = None)
y1 = df1.iloc[:, 1]
y2 = df2.iloc[:, 1]
y1_sum = y1_sum + y1
y2_sum = y2_sum + y2
visualizeFeatureIMP(y1, y2)

plt.legend(loc="center right", fontsize=6, bbox_to_anchor=(0.99, 0.4))
plt.text(x=0, y=0.95, s=' (a) ILL18', weight="bold")
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(2))
ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax1.axes.xaxis.set_ticklabels([])

text_plot = True
if text_plot == True:
    ax1.text(0, 0.75, s="Benford's Law Set",
             ha='left', va='center', fontsize=5, color="black")
    ax1.text(11, 0.75, s="Waveform Set",
             ha='left', va='center', fontsize=5, color="black")
    ax1.text(36, 0.75, s="Spectral Set",
             ha='left', va='center', fontsize=5, color="black")
    ax1.text(53, 0.75, s="Spectrogram Set",
             ha='left', va='center', fontsize=5, color="black")
    ax1.text(70, 0.75, s="Network Set",
             ha='left', va='center', fontsize=5, color="black")

# </editor-fold>


# <editor-fold desc="ILL12">
ax2 = plt.subplot(gs[1])
#ax2.set_title('(b) ILL12', weight="bold", loc='left')

input_station, feature_type = "ILL12", "C"
model_type = "Random_Forest"
df1 = pd.read_csv(f"{parent_dir}/output/train_test_output/figures/"
                  f"{input_station}_{model_type}_{feature_type}_{input_component}_build_in_IMP.txt", header = None)
model_type = "XGBoost"
df2 = pd.read_csv(f"{parent_dir}/output/train_test_output/figures/"
                  f"{input_station}_{model_type}_{feature_type}_{input_component}_build_in_IMP.txt", header = None)
y1 = df1.iloc[:, 1]
y2 = df2.iloc[:, 1]
y1_sum = y1_sum + y1
y2_sum = y2_sum + y2
visualizeFeatureIMP(y1, y2)
plt.text(x=0, y=0.95, s=' (b) ILL12', weight="bold")

ax2.xaxis.set_minor_locator(ticker.MultipleLocator(2))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax2.axes.xaxis.set_ticklabels([])

# </editor-fold>


# <editor-fold desc="ILL13">
ax3 = plt.subplot(gs[2])
#ax3.set_title('(c) ILL13', weight="bold", loc='left')

input_station, feature_type = "ILL13", "C"
model_type = "Random_Forest"
df1 = pd.read_csv(f"{parent_dir}/output/train_test_output/figures/"
                  f"{input_station}_{model_type}_{feature_type}_{input_component}_build_in_IMP.txt", header = None)
model_type = "XGBoost"
df2 = pd.read_csv(f"{parent_dir}/output/train_test_output/figures/"
                  f"{input_station}_{model_type}_{feature_type}_{input_component}_build_in_IMP.txt", header = None)
y1 = df1.iloc[:, 1]
y2 = df2.iloc[:, 1]
y1_sum = y1_sum + y1
y2_sum = y2_sum + y2
visualizeFeatureIMP(y1, y2)
plt.text(x=0, y=0.95, s=' (c) ILL13', weight="bold")

ax3.xaxis.set_minor_locator(ticker.MultipleLocator(2))
ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))


text_plot = True
if text_plot == True:
    ax3.text(36, 0.95, s="ID 0: First digit frequency of 1",
             ha='left', va='center', fontsize=5, color="black")
    ax3.text(36, 0.85, s="ID 8: First digit frequency of 9",
             ha='left', va='center', fontsize=5, color="black")
    ax3.text(36, 0.75, s="ID 10: Power law exponent",
             ha='left', va='center', fontsize=5, color="black")
    ax3.text(36, 0.65, s="ID 14: Kurtosis of the signal",
             ha='left', va='center', fontsize=5, color="black")

    ax3.text(53, 0.95, s="ID 22: Energy of signals filtered in 1-5 Hz",
             ha='left', va='center', fontsize=5, color="black")
    ax3.text(53, 0.85, s="ID 23: Energy of signals filtered in 5-15 Hz",
             ha='left', va='center', fontsize=5, color="black")
    ax3.text(53, 0.75, s="ID 24: Energy of signals filtered in 15-25 Hz",
             ha='left', va='center', fontsize=5, color="black")

    ax3.text(53, 0.65, s="ID 35: Interquartile range",
             ha='left', va='center', fontsize=5, color="black")
    ax3.text(53, 0.55, s="ID 47: Energy between 1/4 and 1/2 Nyquist frequency",
             ha='left', va='center', fontsize=5, color="black")

# </editor-fold>


fig.text(x=0,   y=0.5,  s='Normalized Weight', weight='bold', va='center', rotation='vertical')
fig.text(x=0.45, y=0.01, s="Feature Index (ID)", fontweight="bold")

plt.tight_layout()
plt.subplots_adjust(hspace=0.1, right = 0.98)
plt.savefig(f"{parent_dir}/plotting/feature_imp/feature_imp_based_on_{feature_type}.png", dpi=600)
plt.show()

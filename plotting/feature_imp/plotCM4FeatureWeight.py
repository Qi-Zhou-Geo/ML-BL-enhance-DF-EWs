#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )#, 'font.weight':'bold'
OUTPUT_DIR = "/Users/qizhou/#file/2_projects/FI_paper/1figs/3FeatureWeight/"
DATA_DIR = "/Users/qizhou/#file/2_projects/ML_paper/1figures/#algorithmsTYPEoutputFinal/"


def plot_cm(data_array, f1, index, model, station):
    data_array = data_array.reshape(2, 2)
    data1_normalize = data_array / data_array.sum(axis=1)[:, np.newaxis]
    data1_normalize = data1_normalize.astype(np.float32)

    sns.heatmap(data1_normalize, xticklabels=False, yticklabels=False, annot=True, fmt='.3f',
                square=True, cmap="Blues", cbar=False, annot_kws={"size": 6, "font": "Arial"})

    plt.text(x=0.5, y=0.4, s=f"{data_array[0, 0]}", ha='center', color="white", fontsize=6)
    plt.text(x=1.5, y=0.4, s=f"{data_array[0, 1]}", ha='center', color="black", fontsize=6)

    plt.text(x=0.5, y=1.4, s=f"{data_array[1, 0]}", ha='center', color="black", fontsize=6)
    plt.text(x=1.5, y=1.4, s=f"{data_array[1, 1]}", ha='center', color="white", fontsize=6)

    plt.text(x=0, y=1.9, s=f"F1={f1:.3f}", color="black", fontsize=6, ha="left")
    plt.text(x=0, y=0.15, s=f"{index} {model}-{station}", color="white", fontsize=7, ha="left", weight='bold')

    #plt.ylabel(f"Actual Class", weight='bold')
    plt.yticks([0.5, 1.5], ["", ""], rotation='vertical', ha='center', va='center')

    #plt.xlabel(f"Predicted Class", weight='bold')
    plt.xticks([0.5, 1.5], ["", ""])


## output CM for table
def loopForCM(station, model, featureTYPE):
    if model == "RF" or model == "XGB":
        df = pd.read_csv(f"{DATA_DIR}{model}{featureTYPE}/{station}{model}_classification_testOut2020.txt", header=0)#trainOut2017
    elif model == "LSTM":
        df = pd.read_csv(f"{DATA_DIR}{model}{featureTYPE}/{station}_LSTM_s64b16_testOut2020.txt", header=0)#trainOut2017-2019
    all_targets = df.iloc[:, 2]
    all_outputs = df.iloc[:, 5]

    if station == "ILL13":
        id1 = np.where(np.array(df.iloc[:, 0]) == "2020-06-16 23:50:00")[0][0]
        id2 = np.where(np.array(df.iloc[:, 0]) == "2020-06-17 00:30:00")[0][0]
        all_targets[id1:id2] = 1
    else:
        pass

    cm_raw = confusion_matrix(all_targets, all_outputs).reshape(-1)
    TN, FP, FN, TP = cm_raw[0], cm_raw[1], cm_raw[2], cm_raw[3]
    f1 = f1_score(all_targets, all_outputs, average='binary', zero_division=0)
    TS = TP / (TP + FP + FN)

    print(station, model, featureTYPE, "Negative", TN + FP, "Positive", FN+TP)

    return np.array([TN, FP, FN, TP]), f1, TS


fig = plt.figure(figsize=(5.5, 4))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])


ax0 = plt.subplot(gs[0])
#ax0.set_title('(a) RF model trained with Type A & B features', weight="bold", loc='left')
cm, f1, ts = loopForCM(station="ILL18", model="RF", featureTYPE="bl_rf")
plot_cm(cm, f1, " (a)", "RF", "ILL18")
plt.ylabel(f"Actual Class", weight='bold')
plt.yticks([0.5, 1.5], ["Non-DF", "DF"], rotation='vertical', ha='center', va='center')
#plt.xlabel(f"Predicted Class (ILL18)", weight='bold')
#plt.xticks([0.5, 1.5], ["Non-DF", "DF"])

ax1 = plt.subplot(gs[1])
cm, f1, ts = loopForCM(station="ILL12", model="RF", featureTYPE="bl_rf")
plot_cm(cm, f1,  " (b)", "RF", "ILL12")
#plt.xlabel(f"Predicted Class (ILL12)", weight='bold')
#plt.xticks([0.5, 1.5], ["Non-DF", "DF"])

ax2 = plt.subplot(gs[2])
cm, f1, ts = loopForCM(station="ILL13", model="RF", featureTYPE="bl_rf")
plot_cm(cm, f1,  " (c)", "RF", "ILL13")
#plt.xlabel(f"Predicted Class (ILL3)", weight='bold')
#plt.xticks([0.5, 1.5], ["Non-DF", "DF"])



ax3 = plt.subplot(gs[3])
plt.subplots_adjust(hspace=0.5)
cm, f1, ts = loopForCM(station="ILL18", model="XGB", featureTYPE="bl_rf")
plot_cm(cm, f1,  " (d)", "XGB", "ILL18")
plt.ylabel(f"Actual Class", weight='bold')
plt.yticks([0.5, 1.5], ["Non-DF", "DF"], rotation='vertical', ha='center', va='center')
plt.xlabel(f"Predicted Class", weight='bold')
plt.xticks([0.5, 1.5], ["Non-DF", "DF"])


ax4 = plt.subplot(gs[4])
cm, f1, ts = loopForCM(station="ILL12", model="XGB", featureTYPE="bl_rf")
plot_cm(cm, f1,  " (e)", "XGB", "ILL12")
plt.xlabel(f"Predicted Class", weight='bold')
plt.xticks([0.5, 1.5], ["Non-DF", "DF"])


ax5 = plt.subplot(gs[5])
cm, f1, ts = loopForCM(station="ILL13", model="XGB", featureTYPE="bl_rf")
plot_cm(cm, f1,  " (f)", "XGB", "ILL13")
plt.xlabel(f"Predicted Class", weight='bold')
plt.xticks([0.5, 1.5], ["Non-DF", "DF"])


plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}CM_RF+XGB_bl-rf.png", dpi=600)
plt.show()

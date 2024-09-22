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

# Specify the directory containing the Arial font
from matplotlib import font_manager
font_dirs = ['/storage/vast-gfz-hpc-01/home/qizhou/2python/font']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

# Add fonts to the FontManager
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )#, 'font.weight':'bold'


def plot_cm(data_array, f1, index, model_type, input_station):
    data_array = data_array.reshape(2, 2)
    data1_normalize = data_array / data_array.sum(axis=1)[:, np.newaxis]
    data1_normalize = data1_normalize.astype(np.float32)

    sns.heatmap(data1_normalize, xticklabels=False, yticklabels=False, annot=True, fmt='.3f',
                square=False, cmap="Blues", cbar=False, annot_kws={"size": 6, "font": "Arial"})

    plt.text(x=0.5, y=0.4, s=f"{data_array[0, 0]}", ha='center', color="white", fontsize=6)
    plt.text(x=1.5, y=0.4, s=f"{data_array[0, 1]}", ha='center', color="black", fontsize=6)

    plt.text(x=0.5, y=1.4, s=f"{data_array[1, 0]}", ha='center', color="black", fontsize=6)
    plt.text(x=1.5, y=1.4, s=f"{data_array[1, 1]}", ha='center', color="white", fontsize=6)

    plt.text(x=0, y=1.9, s=f"F1={f1:.3f}", color="black", fontsize=6, ha="left")
    plt.text(x=0, y=0.15, s=f" {index} {model_type} {input_station}", color="white", fontsize=6, ha="left", weight='bold')

    #plt.ylabel(f"Actual Class", weight='bold')
    #plt.yticks([0.5, 1.5], ["", ""], rotation='vertical', ha='center', va='center')

    #plt.xlabel(f"Predicted Class", weight='bold')
    #plt.xticks([0.5, 1.5], ["", ""])
    plt.yticks([0.5, 1.5], ["Non-DF", "DF"], rotation='vertical', ha='center', va='center')
    plt.xticks([0.5, 1.5], ["Non-DF", "DF"])


def loopForCM(input_station, model_type, feature_type, input_component="EHZ", training_or_testing="testing"):

    df = pd.read_csv(f"{parent_dir}/output_results/predicted_results/{input_station}_{model_type}_{feature_type}_{input_component}_{training_or_testing}_output.txt",
                     header=0)

    all_targets = df.iloc[:, 1]
    all_outputs = df.iloc[:, 2]

    cm_raw = confusion_matrix(all_targets, all_outputs).reshape(-1)
    TN, FP, FN, TP = cm_raw[0], cm_raw[1], cm_raw[2], cm_raw[3]
    f1 = f1_score(all_targets, all_outputs, average='binary', zero_division=0)
    TS = TP / (TP + FP + FN)

    print(input_station, model_type, feature_type, "Negative", TN + FP, "Positive", FN+TP)

    return np.array([TN, FP, FN, TP]), f1, TS


def plot_TPR_FPR(ax, input_station, type="testing"):

    marker4model = {"Random_Forest": "^", "XGBoost": "s", "LSTM": "o"}
    color4features = {"A": "#3B75AF", "B": "#EF8636", "C": "#519E3E", "D": "grey"}

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = pd.read_csv(f"{parent_dir}/output_results/summary_{type}.txt", header=None)
    df = np.array(df)

    for model_type in ["Random_Forest", "XGBoost", "LSTM"]:
        for feature_type in ["A", "B", "C"]:

            arr1 = np.where(df[:, 5] == f" {type}")
            arr2 = np.where(df[:, 2] == f" {input_station}")
            arr3 = np.where(df[:, 3] == f" {model_type}")
            arr4 = np.where(df[:, 4] == f" {feature_type}")

            id = np.intersect1d(np.intersect1d(np.intersect1d(arr1, arr2), arr3), arr4)

            filtered_df = df[id, :]

            tpr = filtered_df[0, 14] / (filtered_df[0, 14] + filtered_df[0, 12])
            fpr = filtered_df[0, 10] / (filtered_df[0, 10] + filtered_df[0, 8])

            if filtered_df[0, 20] > 0 :
                edgecolors = "black"
            else:
                edgecolors = "black"

            ax.scatter(x=fpr, y=tpr, alpha=0.6,
                       marker=marker4model.get(model_type),
                       facecolor=color4features.get(feature_type), edgecolors=edgecolors,
                       label=f"{model_type}-{feature_type}", zorder=2)

            #ax.ticklabel_format(axis='x', style='scientific', scilimits=(-2, 3))
            if input_station == "ILL12":
                plt.xlim(0, 0.003)
            else:
                plt.xlim(0, 0.003)

            #plt.xlim(1e-4, 0.003)
            #plt.xscale("log")
            plt.ylim(0.4, 1)
            plt.grid(axis='y', ls="--", lw=0.5, zorder=1)
            plt.grid(axis='x', ls="--", lw=0.5, zorder=1)
            plt.xlabel("False Positive Rate", weight='bold')


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
feature_type = "C"





# <editor-fold desc="plot">
fig = plt.figure(figsize=(5.5, 5.5))
gs = gridspec.GridSpec(3, 3)

ax = plt.subplot(gs[0])
input_station ="ILL18"
plot_TPR_FPR(ax, input_station)
plt.ylabel("True Positive Rate", weight='bold')
plt.text(x=1e-4, y=0.50, s=f" (a) {input_station}", color="black", fontsize=6, ha="left", weight='bold')


ax = plt.subplot(gs[1])
input_station = "ILL12"
plot_TPR_FPR(ax, input_station)
plt.text(x=1e-4, y=0.50, s=f" (b) {input_station}", color="black", fontsize=6, ha="left", weight='bold')


ax = plt.subplot(gs[2])
input_station = "ILL13"
plot_TPR_FPR(ax, input_station)
plt.text(x=1e-4, y=0.50, s=f" (c) {input_station}", color="black", fontsize=6, ha="left", weight='bold')
#plt.legend(fontsize=6, loc=3)

ax0 = plt.subplot(gs[3])
cm, f1, ts = loopForCM(input_station="ILL18", model_type="Random_Forest", feature_type=feature_type )
plot_cm(cm, f1, " (d)", "RF", "ILL18")

plt.ylabel(f"Actual Class", weight='bold')
plt.yticks([0.5, 1.5], ["Non-DF", "DF"], rotation='vertical', ha='center', va='center')


ax1 = plt.subplot(gs[4])
cm, f1, ts = loopForCM(input_station="ILL12", model_type="Random_Forest", feature_type=feature_type )
plot_cm(cm, f1,  " (e)", "RF", "ILL12")


ax2 = plt.subplot(gs[5])
cm, f1, ts = loopForCM(input_station="ILL13", model_type="Random_Forest", feature_type=feature_type )
plot_cm(cm, f1,  " (f)", "RF", "ILL13")


ax3 = plt.subplot(gs[6])
plt.subplots_adjust(hspace=0.5)
cm, f1, ts = loopForCM(input_station="ILL18", model_type="XGBoost", feature_type=feature_type )
plot_cm(cm, f1,  " (g)", "XGB", "ILL18")
plt.ylabel(f"Actual Class", weight='bold')
plt.xlabel(f"Predicted Class", weight='bold')


ax4 = plt.subplot(gs[7])
cm, f1, ts = loopForCM(input_station="ILL12", model_type="XGBoost", feature_type=feature_type )
plot_cm(cm, f1,  " (h)", "XGB", "ILL12")
plt.xlabel(f"Predicted Class", weight='bold')


ax5 = plt.subplot(gs[8])
cm, f1, ts = loopForCM(input_station="ILL13", model_type="XGBoost", feature_type=feature_type )
plot_cm(cm, f1,  " (i)", "XGB", "ILL13")
plt.xlabel(f"Predicted Class", weight='bold')


plt.tight_layout()
plt.savefig(f"{parent_dir}/plotting/tpr_fpr/fpr_tpr_cm.png", dpi=600)
plt.show()
# </editor-fold>





# <editor-fold desc="label">
fig = plt.figure(figsize=(5.5, 5.5))
gs = gridspec.GridSpec(3, 3)


# <editor-fold desc="FPR-TPR">
ax0 = plt.subplot(gs[0])
for idx1, model in enumerate(["Random_Forest", "XGBoost", "LSTM"]):
    for idx2, feature in enumerate(["A", "B", "C"]):

        b = "black"
        ax0.scatter(x=idx2, y=3-idx1, edgecolors=b, s=30, alpha=0.6,
                    marker=marker4model[model],
                    facecolor=color4features[feature])


plt.xlim(-3, 3)
plt.ylim(-0.5, 5.5)


plt.text(x=-0.5, y=3.8, s="Feature Type", fontsize=6, weight="bold")
plt.text(x=-0.1, y=3.3, s="A", fontsize=6)
plt.text(x=1-0.1, y=3.3, s="B", fontsize=6)
plt.text(x=2-0.1, y=3.3, s="C", fontsize=6)

plt.text(x=-1.5, y=1.3, s="Model Type", fontsize=6, ha='right', rotation=90, weight="bold")
plt.text(x=-0.3, y=3-0.1, s="RF", fontsize=6, ha='right')
plt.text(x=-0.3, y=2-0.1, s="XGB", fontsize=6, ha='right')
plt.text(x=-0.3, y=1-0.1, s="LSTM", fontsize=6, ha='right')


plt.xlabel("False Positive Rate", weight='bold')
plt.ylabel("True Positive Rate", weight='bold')


plt.title(f'(a)', weight="bold", loc='left')
# </editor-fold>


ax = plt.subplot(gs[1])
input_station = "ILL12"
plot_TPR_FPR(ax, input_station)
plt.text(x=1e-4, y=0.50, s=f" (b) {input_station}", color="black", fontsize=6, ha="left", weight='bold')


ax = plt.subplot(gs[2])
input_station = "ILL13"
plot_TPR_FPR(ax, input_station)
plt.text(x=1e-4, y=0.50, s=f" (c) {input_station}", color="black", fontsize=6, ha="left", weight='bold')
#plt.legend(fontsize=6, loc=3)

ax0 = plt.subplot(gs[3])
cm, f1, ts = loopForCM(input_station="ILL18", model_type="Random_Forest", feature_type=feature_type )
plot_cm(cm, f1, " (d)", "RF", "ILL18")

plt.ylabel(f"Actual Class", weight='bold')
plt.yticks([0.5, 1.5], ["Non-DF", "DF"], rotation='vertical', ha='center', va='center')


ax1 = plt.subplot(gs[4])
cm, f1, ts = loopForCM(input_station="ILL12", model_type="Random_Forest", feature_type=feature_type )
plot_cm(cm, f1,  " (e)", "RF", "ILL12")


ax2 = plt.subplot(gs[5])
cm, f1, ts = loopForCM(input_station="ILL13", model_type="Random_Forest", feature_type=feature_type )
plot_cm(cm, f1,  " (f)", "RF", "ILL13")


ax3 = plt.subplot(gs[6])
plt.subplots_adjust(hspace=0.5)
cm, f1, ts = loopForCM(input_station="ILL18", model_type="XGBoost", feature_type=feature_type )
plot_cm(cm, f1,  " (g)", "XGB", "ILL18")
plt.ylabel(f"Actual Class", weight='bold')
plt.xlabel(f"Predicted Class", weight='bold')


ax4 = plt.subplot(gs[7])
cm, f1, ts = loopForCM(input_station="ILL12", model_type="XGBoost", feature_type=feature_type )
plot_cm(cm, f1,  " (e)", "XGB", "ILL12")
plt.xlabel(f"Predicted Class", weight='bold')


ax5 = plt.subplot(gs[8])
cm, f1, ts = loopForCM(input_station="ILL13", model_type="XGBoost", feature_type=feature_type )
plot_cm(cm, f1,  " (f)", "XGB", "ILL13")
plt.xlabel(f"Predicted Class", weight='bold')


plt.tight_layout()
plt.savefig(f"{parent_dir}/plotting/tpr_fpr/label.png", dpi=600, transparent=True)
plt.show()
# </editor-fold>


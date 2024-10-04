#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from sklearn.metrics import confusion_matrix, roc_auc_score

# Specify the directory containing the Arial font
from matplotlib import font_manager
font_dirs = ['/storage/vast-gfz-hpc-01/home/qizhou/2python/font']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

# Add fonts to the FontManager
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


plt.rcParams.update( {'font.size':7, 'font.family': "Arial"} )#, 'font.weight':'bold'


def re_classify(target, pro):

    noise2df = np.arange(0.05, 1, 0.05)
    temp_tpr = []
    temp_fpr = []
    temp_distance = []
    temp_auc = []

    for threshold in noise2df:
        temp_pro = np.array(pro)
        temp_target = np.array(target)

        temp_pro[temp_pro >= threshold] = 1 # debris flow
        temp_pro[temp_pro <  threshold] = 0 # noise

        cm_raw = confusion_matrix(temp_target, temp_pro).reshape(-1)
        auc = roc_auc_score(temp_target, temp_pro)
        TN, FP, FN, TP = cm_raw[0], cm_raw[1], cm_raw[2], cm_raw[3]
        tpr = TP / (TP + FN)
        fpr = FP / (FP + TN)

        temp_tpr.append(tpr)
        temp_fpr.append(fpr)
        temp_distance.append( np.sqrt((1-tpr)**2 + (fpr-0)**2) )
        temp_auc.append(auc)

    #temp_distance = np.array(temp_distance)
    #id = np.where(temp_distance == np.min(temp_distance))[0][0]

    temp_auc = np.array(temp_auc)

    id1 = np.where(temp_auc == np.max(temp_auc))[0][0]
    id2 = np.where(noise2df == 0.5)[0][0]

    optimal_t = noise2df[id1]
    auc1 = temp_auc[id1]
    auc2 = temp_auc[id2]

    print(temp_auc)

    return noise2df, temp_tpr, temp_fpr, optimal_t, auc1, auc2


def plot_TPR_FPR(input_station, model_type, feature_type, noise2df, temp_tpr, temp_fpr, optimal_t, idx, auc1, auc2):

    edge_colors = np.where(np.isclose(noise2df, 0.5, atol=0.01), 'red', 'black')
    scatter = plt.scatter(temp_fpr,
                          temp_tpr,
                          c=noise2df,
                          cmap='viridis',
                          vmin=0,
                          vmax=1,
                          edgecolor=edge_colors,
                          zorder=2)

    plt.xlim(-5e-4, 6e-3)
    plt.ylim(0.2, 1.05)

    #plt.plot(noise2df, noise2df, color="black", ls="-", lw=1, zorder=1)
    plt.grid(axis='y', ls="--", lw=0.5, zorder=1)
    plt.grid(axis='x', ls="--", lw=0.5, zorder=1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.text(x=2e-3, y=0.5,
             s=f"optimal threshold t\n"
               f"t={optimal_t:.2f}, AUC={auc1:.4f}\n"
               f"selected threshold t:\n"
               f"t=0.50, AUC={auc2:.4f}",
             fontsize=6)
    if model_type == "Random_Forest":
        model_type = "Random Forest"
    plt.title(f"{model_type}-{feature_type}", fontsize=6, loc="left", weight='bold')
    plt.colorbar(scatter)


fig = plt.figure(figsize=(6.5, 5.6))
gs = gridspec.GridSpec(3, 3)
fig.suptitle('Training', weight='bold')

input_station = "ILL12"
idx = 0
for model_type in ["Random_Forest", "XGBoost", "LSTM"]:
    for feature_type in ["A", "B", "C"]:

        df = pd.read_csv(f"{parent_dir}/output/train_test_output/predicted_results/"
                         f"{input_station}_{model_type}_{feature_type}_EHZ_training_output.txt", header=0)

        target = df.iloc[:, 1]
        pro = df.iloc[:, 3]

        noise2df, temp_tpr, temp_fpr, optimal_t, auc1, auc2 = re_classify(target, pro)
        ax = plt.subplot(gs[idx])
        plot_TPR_FPR(input_station, model_type, feature_type, noise2df, temp_tpr, temp_fpr, optimal_t, idx, auc1, auc2)

        idx += 1

plt.tight_layout()
plt.savefig(f"roc_traning.png", dpi=600)
plt.show()
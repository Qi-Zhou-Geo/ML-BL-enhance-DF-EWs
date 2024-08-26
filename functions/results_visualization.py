#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import f1_score, confusion_matrix

plt.rcParams.update({'font.size': 7})  # , 'font.family': "Arial"})

def visualize_feature_imp(model, input_features_name,
                          data_year, input_station, model_type, feature_type, component):

    fig = plt.figure(figsize=(5.5, 3))
    ax1 = fig.add_subplot(1, 1, 1)

    y = model.feature_importances_
    # plt.plot(y, drawstyle='steps', label=feature_type.upper())
    sns.barplot(x=np.arange(y.size), y=y, label=feature_type.upper())

    featureIDboundary = np.array([[-0.5, 10], [11, 35], [36, 52], [53, 69], [70, 79.5]])
    for step in range(featureIDboundary.shape[0]):
        if step % 2 == 0:
            facecolor = "black"
        else:
            facecolor = "grey"
        plt.axvspan(xmin=featureIDboundary[step, 0], xmax=featureIDboundary[step, 1],
                    ymin=0, ymax=1, alpha=0.2, edgecolor="None", facecolor=facecolor)

    # plot features name
    arr = np.column_stack((input_features_name, y))
    np.savetxt(f"{parent_dir}/output/figures/{input_station}_{model_type}_{feature_type}_featuresIMP.txt",
               arr, fmt='%s', delimiter=',')

    arrSort = np.argsort(arr[:, 1])[::-1]
    xID = [12, 22, 32, 42, 52] + [12, 22, 32, 42, 52]
    yID = [np.nanmax(y) * 0.8, np.nanmax(y) * 0.8, np.nanmax(y) * 0.8, np.nanmax(y) * 0.8, np.nanmax(y) * 0.8,
           np.nanmax(y) * 0.7, np.nanmax(y) * 0.7, np.nanmax(y) * 0.7, np.nanmax(y) * 0.7, np.nanmax(y) * 0.7]
    try:  # some feature_type may do not have 10 features
        for step in range(10):
            plt.text(x=xID[step], y=yID[step], s=f"ID{arrSort[step]}: {arr[arrSort[step], 0]}", fontsize=5)
    except Exception as e:
        print(e)

    plt.xlim(-0.5, 79.5)
    plt.grid(axis='y', ls="--", lw=0.5)
    plt.legend(loc="upper right", fontsize=5)

    plt.xlabel(f"Feature ID, station: {STATION}", weight='bold')
    plt.ylabel('Features Importance', weight='bold')

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.tight_layout()
    plt.savefig(f"{parent_dir}/output/figures/{input_station}_{model_type}_{feature_type}_featuresIMP.png", dpi=600)
    plt.close(fig)


def visualize_confusion_matrix(y_test, pre_y_test_label, training_or_testing,
                               data_year, input_station, model_type, feature_type, component):
    
    cm_raw = confusion_matrix(y_test, pre_y_test_label)
    cm_df_raw = pd.DataFrame(cm_raw, index=["0:Non-DF", "1:DF"], columns=["0:Non-DF", "1:DF"])

    cm_normalize = confusion_matrix(y_test, pre_y_test_label, normalize='true')
    cm_df_normalize = pd.DataFrame(cm_normalize, index=["0:Non-DF", "1:DF"], columns=["0:Non-DF", "1:DF"])

    f1 = f1_score(y_test, pre_y_test_label, average='binary', zero_division=0)

    fig = plt.figure(figsize=(4.5, 4.5))
    sns.heatmap(cm_df_raw, xticklabels=1, yticklabels=1, annot=True, fmt='.0f', square=True, cmap="Blues", cbar=False)

    plt.text(x=0.5, y=0.62, s=f"{cm_df_normalize.iloc[0, 0]:.4f}", color="black", ha="center")
    plt.text(x=1.5, y=0.62, s=f"{cm_df_normalize.iloc[0, 1]:.4f}", color="black", ha="center")

    plt.text(x=0.5, y=1.62, s=f"{cm_df_normalize.iloc[1, 0]:.4f}", color="black", ha="center")
    plt.text(x=1.5, y=1.62, s=f"{cm_df_normalize.iloc[1, 1]:.4f}", color="black", ha="center")

    plt.ylabel("Actual Class", weight='bold')
    plt.xlabel(f"Predicted Class" + "\n" + f"{training_or_testing}, {STATION}, F1={f1:.4}", weight='bold')

    plt.tight_layout()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    plt.savefig(
        f"{parent_dir}/output/figures/{input_station}_{model_type}_{feature_type}_{training_or_testing}_F1_{f1:.4f}.png",
        dpi=600)

    plt.close(fig)

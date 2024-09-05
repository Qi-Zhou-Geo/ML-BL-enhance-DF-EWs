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

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # get the parent path


#df1 = pd.read_csv(f"{parent_dir}output_results/#version1/output_ensemble/summary1.txt", header=None)
#df2 = pd.read_csv(f"{parent_dir}output_results/#version1/output_ensemble/summary2.txt", header=None)
#df = pd.concat([df1, df2], axis=0)

df = pd.read_csv(f"/Users/qizhou/Desktop/summary.txt", header=None)
df = np.array(df)

for type in ["training", "testing"]:
    for input_station in ["ILL18", "ILL12", "ILL13"]:
        for model_type in ["Random_Forest", "XGBoost", "LSTM"]:
            for feature_type in ["A", "B", "C"]:

                arr1 = np.where(df[:, 5] == f" {type}")
                arr2 = np.where(df[:, 2] == f" {input_station}")
                arr3 = np.where(df[:, 3] == f" {model_type}")
                arr4 = np.where(df[:, 4] == f" {feature_type}")

                id = np.intersect1d(np.intersect1d(np.intersect1d(arr1, arr2), arr3), arr4)

                filtered_df = df[id, :]

                print(f"{type}, {input_station}, {model_type}, {feature_type}, "
                      f"{filtered_df[:, 8][0]}, {filtered_df[:, 10][0]},"
                      f"{filtered_df[:, 12][0]}, {filtered_df[:, 14][0]},"
                      f"{filtered_df[:, 16][0]}, {filtered_df[:, 20][0]}")


#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission


import os
import numpy as np
import pandas as pd

path = "/Users/qizhou/Desktop/event_label/"
files = os.listdir(path)
for file in files:
    df0 = pd.read_csv(f"{path}/{file}", header=0)

    year = file[:4]
    station = file[5:10]

    df_label = np.sum(np.array(df0.iloc[:, 2]))
    nondf_label = len(df0) - np.sum(np.array(df0.iloc[:, 2]))
    print(f"{year}, {station}, {df_label}, {nondf_label}")

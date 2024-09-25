#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import pandas as pd
import numpy as np

def fetc_data(data_path, input_station, model_type, feature_type, input_component, data_start, data_end):

    df = pd.read_csv(f"{data_path}{input_station}_{model_type}_{feature_type}_{input_component}_dual_testing_output.txt")

    date = np.array(df.iloc[:, 0])

    id1 = np.where(date == data_start)[0][0]
    id2 = np.where(date == data_end)[0][0] + 1
    pro = np.array(df.iloc[id1:id2, 3])

    pre_y_label = np.array(df.iloc[:, 2])
    time1 = ["2021-07-16 19:16:00",	"2021-08-17 19:10:00", "2021-07-14 20:55:00"]
    time2 = ["2021-07-16 19:45:00", "2021-08-17 19:40:00", "2021-07-14 21:25:00"]

    for step in np.arange(len(time1)):
        id1 = np.where(date == time1[step])[0][0]
        id2 = np.where(date == time2[step])[0][0] + 1
        pre_y_label[id1-60:id2+120] = 2

    false_positive = (pre_y_label == 1).sum()

    return pro, false_positive


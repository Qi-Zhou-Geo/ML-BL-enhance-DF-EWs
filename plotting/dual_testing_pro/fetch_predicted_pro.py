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

    return pro


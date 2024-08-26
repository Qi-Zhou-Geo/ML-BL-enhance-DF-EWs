#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import pandas as pd

def achieve_predicted_results(pre_y_label, pre_y_pro, training_or_testing, split_id,
                              data_year, input_station, model_type, feature_type, component):

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path

    df = pd.read_csv(f"{parent_dir}/data/{data_year}{input_station}_observed_label.txt",
                      header=0, low_memory=False)
    df0 = df.iloc[split_id:, :]

    column_name1 = f"predicted_y_{training_or_testing}_label"
    column_name2 = f"predicted_y_{training_or_testing}_probability"

    df0.set_index(pd.Index(range(len(df0))), inplace=True)
    df1 = pd.DataFrame(pre_y_label.tolist(), columns=[column_name1])
    df2 = pd.DataFrame(pre_y_pro.tolist(), columns=[column_name2])
    df3 = pd.concat([df0, df1, df2], axis=1, ignore_index=True)

    df3.columns = np.concatenate([df0.columns.values, df1.columns.values, df2.columns.values])
    df3.to_csv(
        f"{parent_dir}/output/predicted_results/{input_station}_{model_type}_{feature_type}_{component}_{training_or_testing}_output.txt",
        index=False)


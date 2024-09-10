#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os

import numpy as np
import pandas as pd

def achieve_predicted_results(time_stamps, obs_y_label, pre_y_label, pre_y_pro,
                              input_station, model_type, feature_type, input_component, training_or_testing, num_feats):

    assert len(time_stamps) == len(obs_y_label), f"check the len(time_stamps) == len(obs_y_label)"
    assert len(time_stamps) == len(pre_y_label), f"check the len(time_stamps) == len(pre_y_label)"
    assert len(time_stamps) == len(pre_y_pro), f"check the len(time_stamps) == len(pre_y_pro)"


    array1 = np.array(time_stamps)
    array2 = np.array(obs_y_label)
    array3 = np.array(pre_y_label)
    array4 = np.array(pre_y_pro)

    df = pd.DataFrame({
        'time_window_start': array1,
        'obs_y_label': array2,
        'pre_y_label': array3,
        'pre_y_pro': array4
    })

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path

    df.to_csv(
        f"{parent_dir}/output/predicted_results_{num_feats}/{input_station}_{model_type}_{feature_type}_{input_component}_{training_or_testing}_output.txt",
        index=False)


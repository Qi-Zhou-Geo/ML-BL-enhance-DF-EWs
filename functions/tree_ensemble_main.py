#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import argparse

from dataset2dataloader import *
from results_visualization import *


def main(data_year, input_station, model_type, feature_type, component):
    data_normalize = True
    split_id = 509760

    # load data
    df, input_features_name = select_features(data_year, input_station, feature_type, component)

    # split the df as feature, label, and time stamsp
    X_train, y_train, time_stamps_train, pro_train, \
    X_test,  y_test,  time_stamps_train, pro_test = split_train_test(df, split_id=split_id)

    if data_normalize is True:
        X_train, X_test = input_data_normalize(X_train, X_test)
    else:
        pass

    # pass data to model
    pre_y_train_label, pre_y_train_pro, \
    pre_y_test_label,  pre_y_test_pro, \
    model = ensemble_model(X_train, y_train, X_test, y_test, input_station, model_type, feature_type, component)

    # achieve the results
    achieve_predicted_results(pre_y_train_label, pre_y_train_pro, "train", split_id,
                              data_year, input_station, model_type, feature_type, component)
    achieve_predicted_results(pre_y_test_label, pre_y_test_pro, "test", split_id,
                              data_year, input_station, model_type, feature_type, component)

    # vasulize the results
    visualize_confusion_matrix(y_train, pre_y_train_label, "train",
                               data_year, input_station, model_type, feature_type, component)
    visualize_confusion_matrix(y_test, pre_y_test_label, "test",
                               data_year, input_station, model_type, feature_type, component)

    visualize_feature_imp(model, inputFeaturesNames,
                          data_year, input_station, model_type, feature_type, component)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--data_year", default="2017-2020", type=str, help="input data year")
    parser.add_argument("--input_station", default="ILL12", type=str, help="input station")
    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--component", default="EHZ", type=str, help="seismic component")

    args = parser.parse_args()

    main(args.data_year, args.input_station, args.model_type, args.feature_type, args.component)


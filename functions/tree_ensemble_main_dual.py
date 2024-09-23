#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-09-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys
import argparse


from datetime import datetime
import numpy as np

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir

from functions.dataset2dataloader import *
from functions.results_achive import *
from functions.tree_ensemble_model import *
from functions.check_undetected_events import *

def main(model_type, feature_type, input_component, ref_station, ref_component, input_seis_network, input_station):

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"Start Job {job_id}: UTC+0, {feature_type, input_component, ref_station, input_seis_network, input_station}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

    data_normalize = True

    # load 2017-2019 trainng data to fit the scaler (this is the only purpose)
    input_features_name, X_train, y_train, _, time_stamps_train = select_features(ref_station, feature_type, ref_component, "training", [2017, 2018, 2019])
    # load NEW testing data
    _, X_test,  y_test,  _, time_stamps_test =  select_features(input_station, feature_type, input_component, "dual_testing", [2021])

    if data_normalize is True:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_test = scaler.fit_transform(X_test)
    else:
        pass

    # pass data to model
    pre_y_test_label, pre_y_test_pro = ensemble_model_dual_test(X_test, ref_station, model_type, feature_type, ref_component)

    # achieve the results
    achieve_predicted_results(time_stamps_test, y_test, pre_y_test_label, pre_y_test_pro,
                              input_station, model_type, feature_type, input_component, "dual_testing")


    print(f"End Job: UTC+0, {input_station, model_type, feature_type, input_component}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')

    # shared parameter
    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")
    # reference parameter
    parser.add_argument("--ref_station", default="ILL12", type=str, help="reference trained model")
    parser.add_argument("--ref_component", default="EHZ", type=str, help="seismic input_component")

    # test parameter
    parser.add_argument("--input_seis_network", default="ILL12", type=str, help="input station")
    parser.add_argument("--input_station", default="ILL12", type=str, help="input station")

    args = parser.parse_args()

    main(args.model_type, args.feature_type, args.input_component,
         args.ref_station, args.ref_component, args.input_seis_network, args.input_station)


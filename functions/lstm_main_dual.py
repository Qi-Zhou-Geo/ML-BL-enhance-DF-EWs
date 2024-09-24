#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys
import argparse

import pandas as pd
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir
from functions.dataset2dataloader import *
from functions.lstm_model import *
from functions.check_undetected_events import *
from functions.feature_imp_shap import *



def main(model_type, feature_type, input_component, seq_length, batch_size, ref_station, ref_component, input_seis_network, input_station, input_data_year):

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"Start Job {job_id}: UTC+0, {input_station, model_type, feature_type, input_component, seq_length, batch_size}",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

    data_normalize = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    map_feature_size = {"A":11, "B":69, "C":80}

    # load 2017-2019 trainng data to fit the scaler (this is the only purpose)
    input_features_name, X_train, y_train, time_stamps_train, _ =  select_features(ref_station, feature_type, ref_component, "training", [2017, 2018, 2019])
    # load NEW testing data
    input_features_name, X_test,  y_test,  time_stamps_test,  _ =  select_features(input_station, feature_type, input_component, "dual_testing", [input_data_year])


    if data_normalize is True:
        X_train, X_test = input_data_normalize(X_train, X_test)
        X_train = pd.DataFrame(X_train, columns=input_features_name)
        X_test  = pd.DataFrame(X_test,  columns=input_features_name)
    else:
        pass

    # prepare the dataloader
    train_df = pd.concat([X_train, y_train, time_stamps_train], axis=1, ignore_index=True)
    train_sequences = data2seq(df=train_df, seq_length=seq_length)
    train_dataloader = dataset2dataloader(data_sequences=train_sequences, batch_size=batch_size, training_or_testing="training")

    test_df = pd.concat([X_test, y_test, time_stamps_test], axis=1, ignore_index=True)
    test_sequences = data2seq(df=test_df, seq_length=seq_length)
    test_dataloader = dataset2dataloader(data_sequences=test_sequences, batch_size=batch_size, training_or_testing="dual_testing")


    # load the model
    model_dual_test = lstm_classifier(feature_size=map_feature_size.get(feature_type), device=device)
    load_ckp = f"{CONFIG_dir['output_dir']}/train_test_output/trained_model/" \
               f"{ref_station}_{model_type}_{feature_type}_{ref_component}.pt"
    model_dual_test.load_state_dict(torch.load(load_ckp, map_location=torch.device('cpu')))
    model_dual_test.to(device)

    # do not need this
    optimizer = torch.optim.Adam(model_dual_test.parameters(), lr=0.0001)
    # Define scheduler: Reduce the LR by factor of 0.1 when the metric (like loss) stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    # dual test the model
    dual_test = lstm_train_test(model_dual_test,
                                optimizer,
                                train_dataloader.dataLoader(),
                                test_dataloader.dataLoader(),
                                device,
                                input_station,
                                model_type,
                                feature_type,
                                input_component,
                                scheduler)
    dual_test.dual_testing() # only run one dual test function


    print(f"End Job: UTC+0, {input_station, model_type, feature_type, input_component, seq_length, batch_size}",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')

    # shared parameter
    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")
    parser.add_argument("--seq_length", default=64, type=int, help="Input sequence length")
    parser.add_argument("--batch_size", default=16, type=int, help='Input batch size on each device')

    # reference parameter
    parser.add_argument("--ref_station", default="ILL12", type=str, help="reference trained model")
    parser.add_argument("--ref_component", default="EHZ", type=str, help="seismic input_component")

    # test parameter
    parser.add_argument("--input_seis_network", default="ILL12", type=str, help="input station")
    parser.add_argument("--input_station", default="ILL12", type=str, help="input station")
    parser.add_argument("--input_data_year", default="ILL12", type=str, help="input station")


    args = parser.parse_args()

    main(args.model_type, args.feature_type, args.input_component, args.seq_length, args.batch_size,
         args.ref_station, args.ref_component, args.input_seis_network, args.input_station,
         args.input_data_year)

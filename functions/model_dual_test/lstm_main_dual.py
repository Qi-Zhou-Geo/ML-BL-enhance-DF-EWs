#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import argparse

import pandas as pd
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn


from dataset2dataloader import *
from lstm_model import *
from check_undetected_events import *
from feature_imp_shap import *

def main(input_station, model_type, feature_type, input_component, seq_length, batch_size):

    print(f"Start Job: UTC+0, {input_station, model_type, feature_type, input_component, seq_length, batch_size}",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

    data_normalize = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    map_feature_size = {"A":11, "B":69, "C":80}

    # load data
    input_features_name, X_train, y_train, time_stamps_train, _ = select_features(input_station, feature_type, input_component, "training")
    input_features_name, X_test,  y_test,  time_stamps_test,  _ =  select_features(input_station, feature_type, input_component, "testing")

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
    test_dataloader = dataset2dataloader(data_sequences=test_sequences, batch_size=batch_size, training_or_testing="testing")


    # load model
    train_model = lstm_classifier(feature_size=map_feature_size.get(feature_type), device=device)
    train_model.to(device)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)


    # train and test
    trainer = lstm_train_test(train_model,
                              optimizer,
                              train_dataloader.dataLoader(),
                              test_dataloader.dataLoader(),
                              device,
                              input_station,
                              model_type,
                              feature_type,
                              input_component)
    trainer.activation()

    # summary the results
    summary_results(input_station, model_type, feature_type, input_component, "training")
    summary_results(input_station, model_type, feature_type, input_component, "testing")

    # vasulize the feature importance
    if feature_type == "C":
        pass
        #imp = shap_imp(input_station, model_type, feature_type, input_component, train_dataloader.dataLoader())

        #visualize_feature_imp("shap_value", imp, input_features_name,
                              #input_station, model_type, feature_type, input_component)
    else:
        pass


    print(f"End Job: UTC+0, {input_station, model_type, feature_type, input_component, seq_length, batch_size}",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--input_station", default="ILL12", type=str, help="input station")
    parser.add_argument("--model_type", default="Random_Forest", type=str, help="model type")
    parser.add_argument("--feature_type", default="C", type=str, help="feature type")
    parser.add_argument("--input_component", default="EHZ", type=str, help="seismic input_component")
    
    parser.add_argument("--seq_length", default=64, type=int, help="Input sequence length")
    parser.add_argument("--batch_size", default=16, type=int, help='Input batch size on each device')

    args = parser.parse_args()

    main(args.input_station, args.model_type, args.feature_type, args.input_component, args.seq_length, args.batch_size)

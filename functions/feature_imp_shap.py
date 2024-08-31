#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os

import pandas as pd
import numpy as np

import joblib
import shap

from results_visualization import *
from lstm_model import *


def convert_dataloader2_list(dataloader):

    data_list = []

    for idx, batch_data in enumerate(dataloader):
        input_data = batch_data['features']  # Shape: (batch_size, sequence_length, feature_size)
        data_list.append(input_data)
        #print(idx, input_data.shape)

    return torch.cat(data_list, dim=0)


def shap_tree_explainer(input_station, model_type, feature_type, input_component, background_data, new_data):

    assert model_type == "Random_Forest" or model_type == "XGBoost", f"Please check the model type {model_type}"
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path

    model = joblib.load(
        f"{parent_dir}/output/trained_model/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")

    explainer = shap.TreeExplainer(model, background_data)
    shap_values = explainer.shap_values(new_data)

    if model_type == "Random_Forest":
        # 3D (num_samples, num_features, num_classes) -> 2D (num_samples, num_features)
        shap_values = np.abs(shap_values).mean(axis=2)
    else:
        pass

    # shap_values_exp = shap.Explanation(shap_values, base_values=explainer.expected_value, data=X_train)
    # shap.plots.bar(shap_values_exp, max_display=80)

    imp = np.abs(shap_values).mean(axis=0)

    return imp


def shap_deep_explainer(input_station, model_type, feature_type, input_component, background_data, new_data):

    assert model_type == "LSTM", f"Please check the model type {model_type}"
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    device = "cpu"

    model = lstm_classifier(feature_size=80, device=device)
    model.load_state_dict(torch.load(
        f"{parent_dir}/output/trained_model/{input_station}_{model_type}_{feature_type}_{input_component}.pt",
        map_location=device))
    model.eval()

    # make sure the "background_data" and "new_data with shape "torch.Size([data_num, batch_size, feature_size])"
    explainer = shap.GradientExplainer(model, background_data)
    # output "shap_values" shape is "torch.Size([data_num, batch_size, feature_size, 2])"
    shap_values = explainer.shap_values(new_data)

    shap_values = np.abs(shap_values).sum(axis=(1, 3))
    imp = shap_values.mean(axis=0)

    return imp


def shap_imp(input_station, model_type, feature_type, input_component, background_data, new_data):
    '''
    Parameters
    ----------
    input_station: str, station
    model_type: str, ML model type
    feature_type: str, seismic feature type
    input_component: str, seismic component
    background_data: mulitple types, like train data,
                     data frame for "shap_tree_explainer"
                     or list for "shap_deep_explainer"
    new_data: mulitple types, like test data,
              data frame for "shap_tree_explainer"
              or list for "shap_deep_explainer"

    Returns
            imp: 1D numpy.ndarray with shape feature_size (Type C feature is 80) or x (other type)
    -------

    '''

    if model_type == "Random_Forest" or model_type == "XGBoost":
        imp = shap_tree_explainer(input_station, model_type, feature_type, input_component, background_data, new_data)
    elif model_type == "LSTM":
        # make sure the "background_data" and "new_data with shape "torch.Size([data_num, batch_size, feature_size])"
        background_data = convert_dataloader2_list(background_data)
        new_data = convert_dataloader2_list(new_data)
        imp = shap_deep_explainer(input_station, model_type, feature_type, input_component, background_data, new_data)

    return imp

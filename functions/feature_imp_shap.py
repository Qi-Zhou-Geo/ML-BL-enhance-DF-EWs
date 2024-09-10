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

import torch

from results_visualization import *
from lstm_model import *

def random_select(X_train, model_type, selected_num=1000):
    '''
    Parameters
    ----------
    X_train: numpy_ndarray, or pytorch dataloader

    Returns:
            background_data, new_data
    -------

    '''
    if model_type != "LSTM":
        data_length = X_train.shape[0]
    else:
        data_length = len(X_train)

    np.random.seed(42)
    selected_indices = np.random.choice(data_length, size=selected_num * 2, replace=False)
    id1, id2 = selected_indices[:selected_num], selected_indices[selected_num:]

    if model_type != "LSTM":
        background_data, new_data = X_train[id1, :], X_train[id2, :]
    else:
        background_data, new_data = [], []

        for idx, batch_data in enumerate(X_train):
            input = batch_data['features']

            if idx in id1:
                background_data.append(input)
            elif idx in id2:
                new_data.append(input)
            else:
                pass

        background_data, new_data = torch.cat(background_data, dim=0), torch.cat(new_data, dim=0)


    return background_data, new_data


def shap_tree_explainer(input_station, model_type, feature_type, input_component, background_data, new_data, num_feats):

    assert model_type == "Random_Forest" or model_type == "XGBoost", f"Please check the model type {model_type}"
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path

    model = joblib.load(
        f"{parent_dir}/output/trained_model_{num_feats}/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")

    explainer = shap.TreeExplainer(model, background_data)
    shap_values = explainer.shap_values(new_data, check_additivity=False)

    if model_type == "Random_Forest":
        # 3D (num_samples, num_features, num_classes) -> 2D (num_samples, num_features)
        shap_values = np.abs(shap_values).mean(axis=2)
    else:
        pass

    # shap_values_exp = shap.Explanation(shap_values, base_values=explainer.expected_value, data=X_train)
    # shap.plots.bar(shap_values_exp, max_display=80)

    imp = np.abs(shap_values).mean(axis=0)

    return imp


def shap_gradient_explainer(input_station, model_type, feature_type, input_component, background_data, new_data, num_feats):

    assert model_type == "LSTM", f"Please check the model type {model_type}"
    device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path

    model = lstm_classifier(feature_size=80, device=device)
    model.load_state_dict(torch.load(
        f"{parent_dir}/output/trained_model_{num_feats}/{input_station}_{model_type}_{feature_type}_{input_component}.pt",
        map_location="cpu"))
    model.to(device)
    model.eval()

    # make sure the "background_data" and "new_data with shape "torch.Size([data_num, batch_size, feature_size])"
    explainer = shap.GradientExplainer(model, background_data.to(device))
    # output "shap_values" shape is "torch.Size([data_num, batch_size, feature_size, 2])"
    shap_values = explainer.shap_values(new_data.to(device), check_additivity=False)

    shap_values = np.abs(shap_values.detach().cpu().numpy()).sum(axis=(1, 3))
    imp = shap_values.mean(axis=0)

    return imp


def shap_deep_explainer(input_station, model_type, feature_type, input_component, background_data, new_data, num_feats):

    assert model_type == "LSTM", f"Please check the model type {model_type}"
    device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path

    model = lstm_classifier(feature_size=80, device=device)
    model.load_state_dict(torch.load(
        f"{parent_dir}/output/trained_model_{num_feats}/{input_station}_{model_type}_{feature_type}_{input_component}.pt",
        map_location="cpu"))
    model.to(device)
    model.eval()

    # make sure the "background_data" and "new_data with shape
    # "torch.Size([data_num , batch_size, feature_size])", data_num = selected_num * batch_size
    background_data, new_data = background_data.to(device), new_data.to(device)
    explainer = shap.DeepExplainer(model, background_data)
    # output "shap_values" shape is "torch.Size([data_num, batch_size, feature_size, 2])"
    shap_values = explainer.shap_values(new_data, check_additivity=False)

    shap_values = np.abs(shap_values.detach().cpu().numpy()).sum(axis=(1, 3))
    imp = shap_values.mean(axis=0)


    return imp


def shap_imp(input_station, model_type, feature_type, input_component, data, num_feats):
    '''
    Parameters
    ----------
    input_station: str, station
    model_type: str, ML model type
    feature_type: str, seismic feature type
    input_component: str, seismic component
    data: mulitple types, same as train data,
                     numpy_ndarray for "shap_tree_explainer", or train_dataloader.dataLoader() for shap_deep_explainer

    Returns
            imp: 1D numpy.ndarray with shape feature_size (Type C feature is 80) or x (other type)
    -------

    '''

    if model_type == "Random_Forest" or model_type == "XGBoost":
        background_data, new_data = random_select(data, model_type)
        imp = shap_tree_explainer(input_station, model_type, feature_type, input_component, background_data, new_data, num_feats)
    elif model_type == "LSTM":
        # make sure the "background_data" and "new_data with shape "torch.Size([data_num, batch_size, feature_size])"
        background_data, new_data = random_select(data, model_type)
        imp = shap_deep_explainer(input_station, model_type, feature_type, input_component, background_data, new_data, num_feats)

    return imp

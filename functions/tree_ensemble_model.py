#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir



def ensemble_model(X_train, y_train, X_test, y_test, input_station, model_type, feature_type, input_component):

    model = "define it"
    if model_type == "Random_Forest":
        model = RandomForestClassifier(n_estimators=500)
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=500, importance_type="gain")
    else:
        print(f"check the model_type {model_type}")

    # model training
    model.fit(X_train, y_train)

    pre_y_train_label = model.predict(X_train) # model predicted X_train label
    pre_y_train_pro = model.predict_proba(X_train)[:, 1]  # model predicted X_train probability, column 0 is the pro of Noise
    pre_y_train_pro = np.round(pre_y_train_pro, decimals=3)

    # model testing
    pre_y_test_label = model.predict(X_test)
    pre_y_test_pro = model.predict_proba(X_test)[:, 1]
    pre_y_test_pro = np.round(pre_y_test_pro, decimals=3)

    # save mdoel parameters
    # the first is for reducing feature 1 by 1
    # feature_num = X_train.shape[1] # number of used features
    # joblib.dump(model, f"{CONFIG_dir['output_dir']}/train_test_output/trained_model_{feature_num}/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")
    joblib.dump(model, f"{CONFIG_dir['output_dir']}/train_test_output/trained_model/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")

    # you can load the model as
    # model = joblib.load(f"{parent_dir}/output/trained_model/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")

    return pre_y_train_label, pre_y_train_pro, pre_y_test_label, pre_y_test_pro, model


def ensemble_model_dual_test(X_test, ref_station, model_type, feature_type, ref_component):

    # you can load the reference model as
    model = joblib.load(f"{CONFIG_dir['output_dir']}/train_test_output/trained_model/{ref_station}_{model_type}_{feature_type}_{ref_component}.pkl")

    # model testing
    pre_y_test_label = model.predict(X_test)
    pre_y_test_pro = model.predict_proba(X_test)[:, 1]
    pre_y_test_pro = np.round(pre_y_test_pro, decimals=3)


    return pre_y_test_label, pre_y_test_pro

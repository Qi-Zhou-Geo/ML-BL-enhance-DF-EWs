#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def ensemble_model(X_train, y_train, X_test, y_test, input_station, model_type, feature_type, input_component):

    model = "define it"
    if model_type == "Random_Forest":
        model = RandomForestClassifier(n_estimators=50)
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=200)

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
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path
    joblib.dump(model, f"{parent_dir}/output/trained_model/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")

    # you can load the model as
    # model = joblib.load(f"{parent_dir}/output/{input_station}_{model_type}_{feature_type}_{input_component}.pkl")


    return pre_y_train_label, pre_y_train_pro, pre_y_test_label, pre_y_test_pro, model

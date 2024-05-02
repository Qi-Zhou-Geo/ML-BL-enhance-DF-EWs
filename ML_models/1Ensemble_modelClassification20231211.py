#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
# written by Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# reference to (1) Chmiel, Małgorzata et al. (2021): e2020GL090874, (2) Floriane Provost et al.(2017): 113-120.
# <editor-fold desc="**0** load the package">
import os
import argparse
from datetime import datetime

import pytz
import platform

import numpy as np
import pandas as pd

import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# data visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# </editor-fold>

plt.rcParams.update({'font.size': 7})  # , 'font.family': "Arial"})


# <editor-fold desc="**2** define the input data">

def loadSingleTypeFeatures(STATION, featureTYPE, component="EHZ", dataYear="2017-2020"):
    '''----------
    STATION: load data from which station
    featureTYPE: which types of features
    -------'''
    if featureTYPE == "bl":
        usecols1 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17]  # 15 IQR, 16 goodness, 17 alpha
        usecols2 = [5, 6]  # ration_maxTOminRMS,ration_maxTOminIQR
    elif featureTYPE == "rf":
        usecols1 = np.arange(4, 63)  # load all
        usecols2 = np.arange(3, 13)  # load all

    df1 = pd.read_csv(f"{DATA_DIR}all1component/{dataYear}{STATION}_{component}_{featureTYPE.upper()}.txt",
                      header=0, low_memory=False, usecols=usecols1)
    idFirstNaNrow = df1.notna().idxmax(axis=0)[0]
    values = df1.iloc[idFirstNaNrow, :]
    for step in range(len(values)):
        df1.iloc[:, step].fillna(values[step], inplace=True)
        df1.iloc[:, step].replace([np.inf, -np.inf], values[step], inplace=True)

    df2 = pd.read_csv(f"{DATA_DIR}all1component/{dataYear}{component}net.txt",
                      header=0, low_memory=False, usecols=usecols2)

    # time stamps, binary labels, probability of label 1(DF)
    df3 = pd.read_csv(f"{DATA_DIR}#{dataYear}{STATION}_observedLabels.txt",
                      header=0, low_memory=False,
                      usecols=[1, 2, 4])

    df = pd.concat([df1, df2, df3], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values, df3.columns.values])
    df.columns = columnsName

    inputFeaturesNames = columnsName[:-3]

    return df, inputFeaturesNames


def load_bl_rf_Features(STATION, component="EHZ", dataYear="2017-2020"):
    '''----------
    STATION: load data from which station
    -------'''

    # bl features
    df1 = pd.read_csv(f"{DATA_DIR}all1component/{dataYear}{STATION}_{component}_BL.txt",
                      header=0, low_memory=False, usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17])
    idFirstNaNrow = df1.notna().idxmax(axis=0)[0]
    values = df1.iloc[idFirstNaNrow, :]
    for step in range(len(values)):
        df1.iloc[:, step].fillna(values[step], inplace=True)
        df1.iloc[:, step].replace([np.inf, -np.inf], values[step], inplace=True)

    # rf features
    df2 = pd.read_csv(f"{DATA_DIR}all1component/{dataYear}{STATION}_{component}_RF.txt",
                      header=0, low_memory=False, usecols=np.arange(4, 63))
    idFirstNaNrow = df2.notna().idxmax(axis=0)[0]
    values = df2.iloc[idFirstNaNrow, :]
    for step in range(len(values)):
        df2.iloc[:, step].fillna(values[step], inplace=True)
        df2.iloc[:, step].replace([np.inf, -np.inf], values[step], inplace=True)

    # network features
    df3 = pd.read_csv(f"{DATA_DIR}all1component/{dataYear}{component}net.txt",
                      header=0, low_memory=False, usecols=np.arange(3, 13))

    # time stamps, binary labels, probability of label 1(DF)
    df4 = pd.read_csv(f"{DATA_DIR}#{dataYear}{STATION}_observedLabels.txt",
                      header=0, low_memory=False,
                      usecols=[1, 2, 4])

    df = pd.concat([df1, df2, df3, df4], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values, df3.columns.values, df4.columns.values])
    df.columns = columnsName

    inputFeaturesNames = columnsName[:-3]

    return df, inputFeaturesNames


def load_4Features(STATION, component="EHZ", dataYear="2017-2020"):
    '''
    ----------
    STATION: load data from which station
    -------
    '''

    # bl features, goodness (index in df 16), power law exponent (index in df 17)
    df1 = pd.read_csv(f"{DATA_DIR}all1component/{dataYear}{STATION}_{component}_BL.txt",
                      header=0, low_memory=False, usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12])
    idFirstNaNrow = df1.notna().idxmax(axis=0)[0]
    values = df1.iloc[idFirstNaNrow, :]
    for step in range(len(values)):
        df1.iloc[:, step].fillna(values[step], inplace=True)
        df1.iloc[:, step].replace([np.inf, -np.inf], values[step], inplace=True)

    # rf features
    # 17 ES_2(3-12Hz), 28 IQR
    df2 = pd.read_csv(f"{DATA_DIR}all1component/{dataYear}{STATION}_{component}_RF.txt",
                      header=0, low_memory=False, usecols=[16, 17, 28])
    idFirstNaNrow = df2.notna().idxmax(axis=0)[0]
    values = df2.iloc[idFirstNaNrow, :]
    for step in range(len(values)):
        df2.iloc[:, step].fillna(values[step], inplace=True)
        df2.iloc[:, step].replace([np.inf, -np.inf], values[step], inplace=True)

    # DO NOT use network features

    # time stamps, binary labels, probability of label 1(DF)
    df4 = pd.read_csv(f"{DATA_DIR}#{dataYear}{STATION}_observedLabels.txt",
                      header=0, low_memory=False,
                      usecols=[1, 2, 4])

    df = pd.concat([df1, df2, df4], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values, df4.columns.values])
    df.columns = columnsName

    inputFeaturesNames = columnsName[:-3]

    return df, inputFeaturesNames


def splitDataFrame(df, splitID=509760):
    '''
    Parameters
    ----------
    df: contains all features and labels
    splitID: split the df to train and test data

    Returns: featurts/labels for RF or XGB
    -------

    '''
    train_df = df.iloc[:splitID, :]
    train_df.set_index(pd.Index(range(len(train_df))), inplace=True)
    X_train, y_train, timeStamps_train, pro_train = \
        train_df.iloc[:, :-3], train_df.iloc[:, -2], \
        train_df.iloc[:, -3], train_df.iloc[:, -1]
    X_train, y_train, timeStamps_train, pro_train = \
        X_train.astype(float), y_train.astype(float), \
        timeStamps_train.astype(float), pro_train.astype(float)

    test_df = df.iloc[splitID:, :]
    test_df.set_index(pd.Index(range(len(test_df))), inplace=True)
    X_test, y_test, timeStamps_test, pro_test = \
        test_df.iloc[:, :-3], test_df.iloc[:, -2], \
        test_df.iloc[:, -3], test_df.iloc[:, -1]
    X_test, y_test, timeStamps_test, pro_test = \
        X_test.astype(float), y_test.astype(float), \
        timeStamps_test.astype(float), pro_test.astype(float)

    return X_train, y_train, timeStamps_train, pro_train, \
           X_test, y_test, timeStamps_test, pro_test

# </editor-fold>


# <editor-fold desc="**3** RM and XGB classification model">
def classificationModel(classifier, X_train, y_train, X_test, y_test):
    '''
    Parameters
    ----------
    classifier: RF or XGB classification model
    X_train: features of different time stamps
    y_train: ##labels## of different time stamps
    X_test
    y_test

    Returns: binary labels and probability of different time stamps
    -------
    '''
    # training
    classifier.fit(X_train, y_train)

    y_trainPredictedLabel = classifier.predict(X_train)
    y_trainPredictedProbability = classifier.predict_proba(X_train)[:, 1]  # column 0 is the pro of Noise
    y_trainPredictedProbability = np.round(y_trainPredictedProbability, decimals=3)
    # y_trainPredictedProbability[y_trainPredictedProbability > 1] = 1
    # y_trainPredictedProbability[y_trainPredictedProbability < 0] = 0

    # testing
    y_testPredictedLabel = classifier.predict(X_test)
    y_testPredictedProbability = classifier.predict_proba(X_test)[:, 1]
    y_testPredictedProbability = np.round(y_testPredictedProbability, decimals=3)

    # y_testPredictedProbability[y_testPredictedProbability > 1] = 1
    # y_testPredictedProbability[y_testPredictedProbability < 0] = 0

    # save mdoel
    joblib.dump(classifier, f"{OUTPUT_DIR}{STATION}_{modelTYPE}_{algorithmsTYPE}.pkl")
    # you can load it as
    # model = joblib.load(f"{OUTPUT_DIR}{STATION}_{modelTYPE}_{algorithmsTYPE}.pkl")

    if confusion_matrix(y_train, y_trainPredictedLabel).size != 1:  # only one element in cm
        visualizeConfusionMatrix(y_train, y_trainPredictedLabel, "training")
    if confusion_matrix(y_test, y_testPredictedLabel).size != 1:  # only one element in cm
        visualizeConfusionMatrix(y_test, y_testPredictedLabel, "testing")

    return y_trainPredictedLabel, y_trainPredictedProbability, y_testPredictedLabel, y_testPredictedProbability


def test2013_2014Data(classifier, X_test):
    y_testPredictedLabel = classifier.predict(X_test)
    y_testPredictedProbability = classifier.predict_proba(X_test)[:, 1]

    return y_testPredictedLabel, y_testPredictedProbability


# </editor-fold>


# <editor-fold desc="**4** RF and Regressor model">
def regressorModel(regressor, X_train, y_train, X_test, y_test):
    '''
    Parameters
    ----------
    regressor: RF or XGB regression model
    X_train: features of different time stamps
    y_train: ##probability## of different time stamps
    X_test
    y_test

    Returns: binary labels and probability of different time stamps
    -------
    '''

    # training
    regressor.fit(X_train, y_train)

    y_trainPredictedProbability = regressor.predict(X_train)
    y_trainPredictedLabel = (y_trainPredictedProbability > 0.6).astype(int)
    y_trainPredictedProbability[y_trainPredictedProbability > 1] = 1
    y_trainPredictedProbability[y_trainPredictedProbability < 0] = 0

    # testing
    y_testPredictedProbability = regressor.predict(X_test)
    y_testPredictedLabel = (y_testPredictedProbability > 0.6).astype(int)
    y_testPredictedProbability[y_testPredictedProbability > 1] = 1
    y_testPredictedProbability[y_testPredictedProbability < 0] = 0

    # save mdoel
    joblib.dump(regressor, f"{OUTPUT_DIR}{STATION}_{modelTYPE}_{algorithmsTYPE}.pkl")

    if confusion_matrix((y_train > 0.6).astype(int), y_trainPredictedLabel).size != 1:  # only one element in cm
        visualizeConfusionMatrix((y_train > 0.6).astype(int), y_trainPredictedLabel, "training")
    # if confusion_matrix((y_test> 0.6).astype(int),  y_testPredictedLabel).size != 1: # only one element in cm
    # visualizeConfusionMatrix( (y_test> 0.6).astype(int),  y_testPredictedLabel,  "testing")

    return y_trainPredictedLabel, y_trainPredictedProbability, y_testPredictedLabel, y_testPredictedProbability


# </editor-fold>


# <editor-fold desc="**5** record results">
def recordResults(PredictedLabel, PredictedProbability,
                  training_or_testing, STATION,
                  splitID=509760, dataYear="2017-2020"):

    df = pd.read_csv(f"{DATA_DIR}#{dataYear}{STATION}_observedLabels.txt", header=0, low_memory=False)

    if training_or_testing == "training":
        df0 = df.iloc[:splitID, :]
        df0.set_index(pd.Index(range(len(df0))), inplace=True)
        df1 = pd.DataFrame(PredictedLabel.tolist(), columns=['PredictedLabel'])
        df2 = pd.DataFrame(PredictedProbability.tolist(), columns=['PredictedProbability'])
        dfTrain = pd.concat([df0, df1, df2], axis=1, ignore_index=True)
        dfTrain.columns = np.concatenate([df0.columns.values, df1.columns.values, df2.columns.values])
        if splitID != 509760: # for 2013 data
            dfTrain.to_csv(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_testOut{str(df0.iloc[0, 0])[:4]}.txt", index=False)
        else:
            dfTrain.to_csv(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_trainOut{str(df0.iloc[0, 0])[:4]}.txt", index=False)

    else:
        df0 = df.iloc[splitID:, :]
        df0.set_index(pd.Index(range(len(df0))), inplace=True)
        df1 = pd.DataFrame(PredictedLabel.tolist(), columns=['PredictedLabel'])
        df2 = pd.DataFrame(PredictedProbability.tolist(), columns=['PredictedProbability'])
        dfTest = pd.concat([df0, df1, df2], axis=1, ignore_index=True)
        dfTest.columns = np.concatenate([df0.columns.values, df1.columns.values, df2.columns.values])
        dfTest.to_csv(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_testOut{str(df0.iloc[0, 0])[:4]}.txt", index=False)


def visualizeFeatureIMP(model, inputFeaturesNames, featureTYPE):
    fig = plt.figure(figsize=(5.5, 3))
    ax1 = fig.add_subplot(1, 1, 1)

    y = model.feature_importances_
    # plt.plot(y, drawstyle='steps', label=featureTYPE.upper())
    sns.barplot(x=np.arange(y.size), y=y, label=featureTYPE.upper())

    featureIDboundary = np.array([[-0.5, 10], [11, 35], [36, 52], [53, 69], [70, 79.5]])
    for step in range(featureIDboundary.shape[0]):
        if step % 2 == 0:
            facecolor = "black"
        else:
            facecolor = "grey"
        plt.axvspan(xmin=featureIDboundary[step, 0], xmax=featureIDboundary[step, 1],
                    ymin=0, ymax=1, alpha=0.2, edgecolor="None", facecolor=facecolor)

    # plot features name
    arr = np.column_stack((inputFeaturesNames, y))
    np.savetxt(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_featuresIMP.txt",
               arr, fmt='%s', delimiter=',')

    arrSort = np.argsort(arr[:, 1])[::-1]
    xID = [12, 22, 32, 42, 52] + [12, 22, 32, 42, 52]
    yID = [np.nanmax(y) * 0.8, np.nanmax(y) * 0.8, np.nanmax(y) * 0.8, np.nanmax(y) * 0.8, np.nanmax(y) * 0.8,
           np.nanmax(y) * 0.7, np.nanmax(y) * 0.7, np.nanmax(y) * 0.7, np.nanmax(y) * 0.7, np.nanmax(y) * 0.7]
    try:  # some featureTYPE may do not have 10 features
        for step in range(10):
            plt.text(x=xID[step], y=yID[step], s=f"ID{arrSort[step]}: {arr[arrSort[step], 0]}", fontsize=5)
    except Exception as e:
        print(e)

    plt.xlim(-0.5, 79.5)
    plt.grid(axis='y', ls="--", lw=0.5)
    plt.legend(loc="upper right", fontsize=5)

    plt.xlabel(f"Feature ID, station: {STATION}", weight='bold')
    plt.ylabel('Features Importance', weight='bold')

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_featuresIMP.png", dpi=600)
    plt.show()
    plt.close(fig)


def featureIMPvisualize2(model, inputFeaturesNames, featureTYPE):
    plt.rcParams.update({'font.size': 8})  # , 'font.family': "Arial"})
    feature_imp = pd.Series(model.feature_importances_, index=inputFeaturesNames)

    df = pd.DataFrame({'Feature names': inputFeaturesNames, 'Weights': feature_imp})
    df["station"] = STATION
    df["id"] = np.arange(0, len(df))
    df.to_csv(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_featuresIMP.txt", index=False)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 1, 1)

    if featureTYPE == "rf" or featureTYPE == "bl_rf":
        sns.barplot(x=np.arange(0, len(df['Weights']), 1), y=df['Weights'])
        df_sort = df.sort_values(by='Weights', ascending=False)
        for i in range(0, 5):
            plt.text(x=14 * i, y=max(df['Weights']) * 0.8, s=f"{df_sort.iloc[i, -1]}: {df_sort.iloc[i, 0]}")
        for i in range(5, 10):
            plt.text(x=14 * (i - 5), y=max(df['Weights']) * 0.7, s=f"{df_sort.iloc[i, -1]}: {df_sort.iloc[i, 0]}")
        for i in range(10, 15):
            plt.text(x=14 * (i - 10), y=max(df['Weights']) * 0.6, s=f"{df_sort.iloc[i, -1]}: {df_sort.iloc[i, 0]}")
        for i in range(15, 20):
            plt.text(x=14 * (i - 15), y=max(df['Weights']) * 0.5, s=f"{df_sort.iloc[i, -1]}: {df_sort.iloc[i, 0]}")

        plt.axhline(y=1 / len(feature_imp), color="red", lw=1, ls="--")
        plt.axvline(x=1, color="red", lw=1, ls="--")
        plt.axvline(x=10, color="red", lw=1, ls="--")
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))

    elif featureTYPE == "bl":
        sns.barplot(x=df['Feature names'], y=df['Weights'])
        plt.axhline(y=1 / len(feature_imp), color="red", lw=1, ls="--")

    elif featureTYPE == "ALL_rf":
        sns.barplot(x=np.arange(0, len(df['Weights']), 1), y=df['Weights'])
        df_sort = df.sort_values(by='Weights', ascending=False)
        for i in range(0, 5):
            plt.text(x=30 * i, y=max(df['Weights']) * 0.8, s=f"{df_sort.iloc[i, -1]}: {df_sort.iloc[i, 0]}")
        for i in range(5, 10):
            plt.text(x=30 * (i - 5), y=max(df['Weights']) * 0.7, s=f"{df_sort.iloc[i, -1]}: {df_sort.iloc[i, 0]}")
        for i in range(10, 15):
            plt.text(x=30 * (i - 10), y=max(df['Weights']) * 0.6, s=f"{df_sort.iloc[i, -1]}: {df_sort.iloc[i, 0]}")
        for i in range(15, 20):
            plt.text(x=30 * (i - 15), y=max(df['Weights']) * 0.5, s=f"{df_sort.iloc[i, -1]}: {df_sort.iloc[i, 0]}")

        plt.axhline(y=1 / len(feature_imp), color="red", lw=1, ls="--")
        plt.axvline(x=8, color="red", lw=1, ls="--")
        plt.axvline(x=68, color="red", lw=1, ls="--")
        plt.axvline(x=68 + 60, color="red", lw=1, ls="--")
        plt.axvline(x=68 + 60 * 2, color="red", lw=1, ls="--")

        ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # Add labels to your graph
    plt.xlabel(f"Feature ID, station: {STATION}", weight='bold')
    plt.ylabel('Features Importance', weight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_featuresIMP.png", dpi=600)
    plt.close(fig)


def visualizeConfusionMatrix(all_targets, all_outputs, training_or_testing):
    cm_raw = confusion_matrix(all_targets, all_outputs)
    cm_df_raw = pd.DataFrame(cm_raw, index=["0:None DF", "1:DF"], columns=["0:None DF", "1:DF"])

    cm_normalize = confusion_matrix(all_targets, all_outputs, normalize='true')
    cm_df_normalize = pd.DataFrame(cm_normalize, index=["0:None DF", "1:DF"], columns=["0:None DF", "1:DF"])

    f1 = f1_score(all_targets, all_outputs, average='binary', zero_division=0)

    fig = plt.figure(figsize=(4.5, 4.5))
    sns.heatmap(cm_df_raw, xticklabels=1, yticklabels=1, annot=True, fmt='.0f', square=True, cmap="Blues", cbar=False)

    plt.text(x=0.5, y=0.62, s=f"{cm_df_normalize.iloc[0, 0]:.4f}", color="black", ha="center")
    plt.text(x=1.5, y=0.62, s=f"{cm_df_normalize.iloc[0, 1]:.4f}", color="black", ha="center")

    plt.text(x=0.5, y=1.62, s=f"{cm_df_normalize.iloc[1, 0]:.4f}", color="black", ha="center")
    plt.text(x=1.5, y=1.62, s=f"{cm_df_normalize.iloc[1, 1]:.4f}", color="black", ha="center")

    plt.ylabel("Actual Class", weight='bold')
    plt.xlabel(f"Predicted Class" + "\n" + f"{training_or_testing}, {STATION}, F1={f1:.4}", weight='bold')

    plt.tight_layout()
    if training_or_testing == "training":
        plt.savefig(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_trainingF1_{f1:.4f}.png", dpi=600)
    elif training_or_testing == "testing":
        plt.savefig(f"{OUTPUT_DIR}{STATION}{modelTYPE}_{algorithmsTYPE}_testingF1_{f1:.4f}.png", dpi=600)
    plt.close(fig)


# </editor-fold>



# <editor-fold desc="**6** cal function">
def runType1(STATION, featureTYPE, modelTYPE, scalerData=False):
    '''
    Parameters
    ----------
    STATION: data source
    featureTYPE: feature type
    modelTYPE: model type
    algorithmsTYPE: algorithm type

    Returns
    -------
    '''

    scaler = MinMaxScaler()

    # <editor-fold desc="load data">
    if featureTYPE == "bl_rf":
        df, inputFeaturesNames = load_bl_rf_Features(STATION)
    elif featureTYPE == "bl" or featureTYPE == "rf":
        df, inputFeaturesNames = loadSingleTypeFeatures(STATION, featureTYPE)
    elif featureTYPE == "4features":
        df, inputFeaturesNames = load_4Features(STATION)
    else:
        print("check the featureTYPE")

    X_train, y_train, timeStamps_train, pro_train, \
    X_test, y_test, timeStamps_test, pro_test = splitDataFrame(df, splitID=509760)

    if scalerData == True:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    # </editor-fold>

    # <editor-fold desc="load model and train it">
    model = "define it"
    if modelTYPE == "RF":
        model = RandomForestClassifier(n_estimators=800)
    elif modelTYPE == "XGB":
        model = XGBClassifier(n_estimators=200)

    y_trainPredictedLabel, y_trainPredictedProbability, \
    y_testPredictedLabel, y_testPredictedProbability = \
        classificationModel(model, X_train, y_train, X_test, y_test)
    # </editor-fold>

    # <editor-fold desc="visualize the results">
    recordResults(y_trainPredictedLabel, y_trainPredictedProbability, "training", STATION,
                  splitID=509760, dataYear="2017-2020")
    recordResults(y_testPredictedLabel, y_testPredictedProbability, "testing", STATION,
                  splitID=509760, dataYear="2017-2020")
    visualizeFeatureIMP(model, inputFeaturesNames, featureTYPE)
    # </editor-fold>

    # <editor-fold desc="load 2013-2014 data">
    stationMapping = {'ILL18':'IGB01', 'ILL12':'IGB02', 'ILL13':'IGB10'}
    if featureTYPE == "bl_rf":
        df2013_2014, inputFeaturesNames = load_bl_rf_Features(stationMapping.get(STATION), "BHZ", "2013-2014")
    elif featureTYPE == "bl" or featureTYPE == "rf":
        df2013_2014, inputFeaturesNames = loadSingleTypeFeatures(stationMapping.get(STATION), featureTYPE, "BHZ", "2013-2014")
    elif featureTYPE == "4features":
        df2013_2014, inputFeaturesNames = load_4Features(stationMapping.get(STATION), "BHZ", "2013-2014")
    else:
        print("check the featureTYPE")

    X_test2013, y_train, timeStamps_train, pro_train, \
    X_test2014, y_test, timeStamps_test, pro_test = splitDataFrame(df2013_2014, splitID=102241)

    if scalerData == True:
        X_test2013 = scaler.transform(X_test2013) # 2013 data
        X_test2014 = scaler.transform(X_test2014) # 2014 data

    # test 2013 and 2014 data
    y_testPredictedLabel, y_testPredictedProbability = test2013_2014Data(model, X_test2013)
    recordResults(y_testPredictedLabel, y_testPredictedProbability, "training", stationMapping.get(STATION),
                  splitID=102241, dataYear="2013-2014")

    y_testPredictedLabel, y_testPredictedProbability = test2013_2014Data(model, X_test2014)
    recordResults(y_testPredictedLabel, y_testPredictedProbability, "testing", stationMapping.get(STATION),
                  splitID=102241, dataYear="2013-2014")
    # </editor-fold>

# </editor-fold>


def main(input_station: str, bl_or_rf: str, RF_or_XGB: str, classificationORregression: str):
    global STATION, DATA_DIR, OUTPUT_DIR, featureTYPE, modelTYPE, algorithmsTYPE

    STATION = input_station
    featureTYPE = bl_or_rf
    modelTYPE = RF_or_XGB
    algorithmsTYPE = classificationORregression
    DATA_DIR = f"/home/qizhou/1projects/dataForML/out60finish/processed/"
    OUTPUT_DIR = f"/home/qizhou/3paper/2AGU/1featureIMP/{modelTYPE}{featureTYPE}/"

    print(STATION, featureTYPE, modelTYPE, algorithmsTYPE, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


    runType1(STATION, featureTYPE, modelTYPE, scalerData=True)

    print(STATION, featureTYPE, modelTYPE, algorithmsTYPE, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed job')

    parser.add_argument("--input_station", default="ILL12", type=str, help="input station")
    parser.add_argument("--bl_or_rf", default="RF", type=str, help="feature type")
    parser.add_argument("--RF_or_XGB", default="RF", type=str, help="model type")
    parser.add_argument("--classificationORregression", default="RF", type=str, help="model type")

    args = parser.parse_args()

    main(args.input_station, args.bl_or_rf, args.RF_or_XGB, args.classificationORregression)


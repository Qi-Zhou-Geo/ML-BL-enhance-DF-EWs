#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
import numpy as np


def load_all_features(data_year, input_station, component):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # get the parent path

    # BL sets
    df1 = pd.read_csv(f"{parent_dir}/data/{data_year}{input_station}_{component}_A.txt",
                      header=0, low_memory=False, usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17])
    idFirstNaNrow = df1.notna().idxmax(axis=0)[0]
    values = df1.iloc[idFirstNaNrow, :]
    for step in range(len(values)):
        df1.iloc[:, step].fillna(values[step], inplace=True)
        df1.iloc[:, step].replace([np.inf, -np.inf], values[step], inplace=True)

    # waveform, spectrum, spectrogram sets
    df2 = pd.read_csv(f"{parent_dir}/data/{data_year}{input_station}_{component}_B.txt",
                      header=0, low_memory=False, usecols=np.arange(4, 63))
    idFirstNaNrow = df2.notna().idxmax(axis=0)[0]
    values = df2.iloc[idFirstNaNrow, :]
    for step in range(len(values)):
        df2.iloc[:, step].fillna(values[step], inplace=True)
        df2.iloc[:, step].replace([np.inf, -np.inf], values[step], inplace=True)

    # network sets
    df3 = pd.read_csv(f"{parent_dir}/data/{data_year}{component}_net.txt",
                      header=0, low_memory=False, usecols=np.arange(3, 13))

    # time stamps, binary labels, probability of label 1(DF)
    df4 = pd.read_csv(f"{parent_dir}/data/{data_year}{input_station}_observed_label.txt",
                      header=0, low_memory=False,
                      usecols=[1, 2, 4])

    df = pd.concat([df1, df2, df3, df4], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values, df3.columns.values, df4.columns.values])
    df.columns = columnsName

    return df


def select_features(data_year, input_station, feature_type, component):
    df = load_all_features(data_year, input_station, component)

    if feature_type is "A": # BL sets
        selected_column = np.arange(0, 11).tolist()
        selected_column.extend([80, 81, 82])
    elif feature_type is "B": # waveform, spectrum, spectrogram, and network sets
        selected_column = np.arange(11, 83).tolist()
    elif feature_type is "C": # A and B
        selected_column = np.arange(0, 83).tolist()
    elif feature_type is "D": # selected from C
        selected_column = [0, 8, 10, 23, 24, 35]
    else:
        print(f"please check the {feature_type}")

    df = df.iloc[:, selected_column]
    input_features_name = df.columns[:-3]

    return df, input_features_name


def split_train_test(df, split_id):
    train_df = df.iloc[:split_id, :]
    train_df.set_index(pd.Index(range(len(train_df))), inplace=True)
    X_train, y_train, time_stamps_train, pro_train = \
        train_df.iloc[:, :-3], train_df.iloc[:, -2], \
        train_df.iloc[:, -3], train_df.iloc[:, -1]
    X_train, y_train, time_stamps_train, pro_train = \
        X_train.astype(float), y_train.astype(float), \
        time_stamps_train.astype(float), pro_train.astype(float)

    test_df = df.iloc[split_id:, :]
    test_df.set_index(pd.Index(range(len(test_df))), inplace=True)
    X_test, y_test, time_stamps_test, pro_test = \
        test_df.iloc[:, :-3], test_df.iloc[:, -2], \
        test_df.iloc[:, -3], test_df.iloc[:, -1]
    X_test, y_test, time_stamps_test, pro_test = \
        X_test.astype(float), y_test.astype(float), \
        time_stamps_test.astype(float), pro_test.astype(float)

    return X_train, y_train, time_stamps_train, pro_train, \
           X_test, y_test, time_stamps_test, pro_test


def input_data_normalize(X_train, X_test):

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test



def data2seq(df, seq_length):
    '''
    Parameters
    ----------
    df: data frame with features and labels
    seq_length: sequence length

    Returns
    -------
    '''

    df_array = df.values.astype('float64')  # Convert DataFrame to NumPy array
    sequences = []

    for i in range(len(df_array) - seq_length - 1):
        features = df_array[i:i + seq_length, :-3]

        time_stamps = df_array[i + seq_length, -3]
        label = df_array[i + seq_length, -2]
        probability = df_array[i + seq_length, -1]

        sequences.append((features, time_stamps, label, probability))

    return sequences


class seq2dataset(Dataset):
    # sequence to dataset
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        inputFeatures, time_stamps, label, probability = self.sequences[index]
        return dict(
            features=torch.Tensor(inputFeatures).to(torch.float64),

            timestamps=torch.tensor(time_stamps).to(torch.float64),
            label=torch.tensor(label).to(torch.long),  # other type will raise error in loss function
            pro=torch.tensor(probability).to(torch.float64)
        )


class dataset2dataloader:
    # dataset to dataloader
    def __init__(self, data_sequences, batch_size, training_or_testing):
        self.data_sequences = data_sequences
        self.batch_size = batch_size
        self.training_or_testing = training_or_testing
        self.data_dataset = seq2dataset(self.data_sequences)

    def dataLoader(self):  # shuffle=False for multiple GPU
        # num_workers=2 for node 501-502
        if self.training_or_testing == "training":  # sampler=DistributedSampler(self.data_dataset) for multiple-gpu
            dataLoader = DataLoader(self.data_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=True, pin_memory=True, num_workers=2)  # , sampler=DistributedSampler(self.data_dataset))
        elif self.training_or_testing == "testing":  # shuffle=False for validation and test, the num_cpu=4,  Do NOT use sampler=DistributedSampler
            dataLoader = DataLoader(self.data_dataset, batch_size=self.batch_size,
                                    shuffle=False, drop_last=True, pin_memory=True, num_workers=2)
        return dataLoader


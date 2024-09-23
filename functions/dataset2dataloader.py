#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import CONFIG_dir as a global variable
from config.config_dir import CONFIG_dir


def load_all_features(input_year, input_station, input_component):

    # BL sets
    df1 = pd.read_csv(f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}_{input_station}_{input_component}_all_A.txt",
                      header=0, low_memory=False, usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17])

    # waveform, spectrum, spectrogram sets
    df2 = pd.read_csv(f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}_{input_station}_{input_component}_all_B.txt",
                      header=0, low_memory=False, usecols=np.arange(4, 63))

    # network sets
    df3 = pd.read_csv(f"{CONFIG_dir['output_dir']}/data_output/seismic_feature/{input_year}_{input_component}_all_network.txt",
                      header=0, low_memory=False, usecols=np.arange(3, 13))

    # time stamps, binary labels, probability of label 1(DF)
    df4 = pd.read_csv(f"{parent_dir}/data/event_label/{input_year}_{input_station}_{input_component}_observed_label.txt",
                      header=0, low_memory=False)

    df = pd.concat([df1, df2, df3, df4], axis=1, ignore_index=True)
    columnsName = np.concatenate([df1.columns.values, df2.columns.values, df3.columns.values, df4.columns.values])
    df.columns = columnsName

    return df


def select_features(input_station, feature_type, input_component, training_or_testing):

    mapping_input_station = {"ILL18":"ILL08", "ILL12":"ILL02", "ILL13":"ILL03"}

    if training_or_testing == "training":
        input_year_data = [2017, 2018, 2019]
    elif training_or_testing == "testing":
        input_year_data = [2020]
    else:
        print(f"please check the training_or_testing, {training_or_testing}")


    df = pd.DataFrame()
    for input_year in input_year_data:
        if input_year == 2017:
            station = mapping_input_station.get(input_station)
        else:
            station = input_station
        df1 = load_all_features(input_year, station, input_component)

        df = pd.concat([df, df1], axis=0, ignore_index=True)

    if feature_type == "A": # BL sets
        selected_column = np.arange(0, 11).tolist()
        selected_column.extend([80, 81, 82])
    elif feature_type == "B": # waveform, spectrum, spectrogram, and network sets
        selected_column = np.arange(11, 83).tolist()
    elif feature_type == "C": # A and B
        selected_column = np.arange(0, 83).tolist()
    elif feature_type == "D": # selected from C
        selected_column = [0, 8, 10, 23, 24, 35, 80, 81, 82]
    else:
        print(f"please check the {feature_type}")

    df = df.iloc[:, selected_column]
    input_features_name = df.columns[:-3]
    
    
    x, y, time_stamp_float, time_stamp_str = df.iloc[:, :-3], df.iloc[:, -1], df.iloc[:, -2], df.iloc[:, -3]

    return input_features_name, x, y, time_stamp_float, time_stamp_str

# Assume C features are input
def select_specific_features(x, feature_names, selected_idxs):
    """
    Select specific features from the input data based on the given indices.

    Parameters:
    x : array-like, shape (n_samples, n_features)
        The input data from which features need to be selected.
    
    feature_names : list of str
        The list of all feature names corresponding to the columns of x.
    
    selected_idxs : list of int
        The indices of the features to be selected.
    
    Returns:
    x_selected : array-like, shape (n_samples, len(selected_idxs))
        The data containing only the selected features.
    
    selected_feature_names : list of str
        The names of the selected features.
    
    Raises:
    IndexError: If any index in selected_idxs is out of bounds for the given feature_names or x.
    """
    
    # Check that the number of features matches the length of feature_names
    if x.shape[1] != len(feature_names):
        raise ValueError(f"The number of features in x {x.shape[1]} does not match the length of feature_names {len(feature_names)}.")
    
    # Validate indices
    for idx in selected_idxs:
        if idx < 0 or idx >= len(feature_names):
            raise IndexError(f"Index {idx} is out of bounds for the feature_names list with length {len(feature_names)}.")
    
    # Extract the selected features from x using the given indices
    try:
        x_selected = x.iloc[:, selected_idxs]
    except IndexError as e:
        raise IndexError(f"Error selecting indices {selected_idxs} from x with shape {x.shape}: {e}")
    
    # Extract the names of the selected features
    selected_feature_names = [feature_names[i] for i in selected_idxs]
    
    return x_selected, selected_feature_names


def input_data_normalize(X_train, X_test):

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def data2seq(df, seq_length):

    df_array = df.values.astype('float64')  # Convert DataFrame to NumPy array
    sequences = []

    for i in range(len(df_array) - seq_length - 1):

        features = df_array[i : i + seq_length, :-2]
        label =    df_array[i + seq_length, -2]
        time_stamps = df_array[i + seq_length, -1]

        sequences.append((features, label, time_stamps))

    return sequences


class seq2dataset(Dataset):
    # sequence to dataset
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        features, label, time_stamps = self.sequences[index]
        return dict(
            features=torch.Tensor(features),
            label=torch.tensor(label).to(torch.long),  # other type will raise error in loss function
            timestamps=torch.tensor(time_stamps)
        )


class dataset2dataloader:
    # dataset to dataloader
    def __init__(self, data_sequences, batch_size, training_or_testing):
        self.data_sequences = data_sequences
        self.batch_size = batch_size
        self.training_or_testing = training_or_testing
        self.dataset = seq2dataset(self.data_sequences)

    def dataLoader(self):

        if self.training_or_testing == "training":
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        elif self.training_or_testing == "testing":
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        else:
            raise ValueError("training_or_testing must be 'training' or 'testing'")

        return data_loader
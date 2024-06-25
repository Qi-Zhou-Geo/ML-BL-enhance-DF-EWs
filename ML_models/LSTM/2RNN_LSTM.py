#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
# written by Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# <editor-fold desc="**** load the package">
import os
import argparse
import platform

import pandas as pd
import numpy as np

import pytz
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns


from tqdm.auto import tqdm
import torch.optim as optim
from torchmetrics import Accuracy
# </editor-fold>


# <editor-fold desc="**0** check the GPU capacity and set GPU id">
def check_cuda_usability():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {device_count}", flush=True)
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")
            print(f"GPU {i}: {torch.cuda.get_device_name(device)}", flush=True)
            print(f"Memory Usage - GPU {i}: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB / "
                  f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB (Allocated / Max Allocated)", flush=True)
            print(f"Memory Cached - GPU {i}: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB / "
                  f"{torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB (Cached / Max Cached)", flush=True)
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available. Using CPU.", flush=True)

    print("CUDA Infor. checked at: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n", flush=True)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def check_data_storage(dataLoader, type):
    iterator = iter(dataLoader)
    batch = next(iterator)
    input = batch['features']
    print("Input feature shape", input.shape, flush=True)

    if str(batch['features'].device).lower() == "cpu":
        print(type, "dataloader stored location:", str(batch['features'].device).lower(), flush=True)
    else:
        print(type, "dataloader stored location", str(batch['features'].device).lower(), flush=True)
    del iterator, batch
# </editor-fold>


# <editor-fold desc="**2** define the input data">

def load_A_or_B_Features(STATION, featureTYPE, component="EHZ", dataYear="2017-2020"):
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

    return df


def load_C_Features(STATION, component="EHZ", dataYear="2017-2020"):
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


    return df


def load_D_Features(STATION, component="EHZ", dataYear="2017-2020"):
    '''----------
    STATION: load data from which station
    -------'''

    # bl features, digit frequency (index in df 4), goodness (index in df 16), power law exponent (index in df 17)
    df1 = pd.read_csv(f"{DATA_DIR}all1component/{dataYear}{STATION}_{component}_BL.txt",
                      header=0, low_memory=False, usecols=[4, 12, 17])
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

    return df

def splitDataFrame(df, splitID = 509760):
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

    test_df = df.iloc[splitID:, :]
    test_df.set_index(pd.Index(range(len(test_df))), inplace=True)

    return train_df, test_df

def DataToSeq(df, seq_length):
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

        timeStamps = df_array[i + seq_length, -3]
        label = df_array[i + seq_length, -2]
        probability = df_array[i + seq_length, -1]

        sequences.append((features, timeStamps, label, probability))

    return sequences

class SeqToDataset(Dataset):
    # sequence to dataset
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        inputFeatures, timeStamps, label, probability = self.sequences[index]
        return dict(
            features=torch.Tensor(inputFeatures).to(torch.float64),

            timestamps=torch.tensor(timeStamps).to(torch.float64),
            label=torch.tensor(label).to(torch.long),  # other type will raise error in loss function
            pro=torch.tensor(probability).to(torch.float64)
        )


class DatasetToDataloader:
    # dataset to dataloader
    def __init__(self, data_sequences, batch_size, training_or_testing):
        self.data_sequences = data_sequences
        self.batch_size = batch_size
        self.training_or_testing = training_or_testing
        self.data_dataset = SeqToDataset(self.data_sequences)

    def dataLoader(self):  # shuffle=False for multiple GPU
        # num_workers=2 for node 501-502
        if self.training_or_testing == "training":  # sampler=DistributedSampler(self.data_dataset) for multiple-gpu
            dataLoader = DataLoader(self.data_dataset, batch_size=self.batch_size,
                                    shuffle=True, drop_last=True, pin_memory=True, num_workers=2)  # , sampler=DistributedSampler(self.data_dataset))
        elif self.training_or_testing == "testing":  # shuffle=False for validation and test, the num_cpu=4,  Do NOT use sampler=DistributedSampler
            dataLoader = DataLoader(self.data_dataset, batch_size=self.batch_size,
                                    shuffle=False, drop_last=True, pin_memory=True, num_workers=2)
        return dataLoader
# </editor-fold>



# <editor-fold desc="**3** define the LSTM structrue">
class LstmClassifier(nn.Module):
    def __init__(self, feature_size, dropout, num_hidden=256, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=num_hidden,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False).to(torch.float64)
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.fully_connect = nn.Linear(num_hidden, output_dim).to(torch.float64)

    def forward(self, x):
        self.lstm.flatten_parameters()
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64).to(DEVICE)

        out, (h_n, c_n) = self.lstm(x, (h0, c0)) # do not need (h_n, c_n)
        # x.shape = [batchSize, sequenceLEN, featureSIZE]
        # out.shape = [batchSize, sequenceLEN, num_hidden]
        # h_n.shape = c_n.shape = [num_layers, batchSize, num_hidden]
        out = out[:, -1, :]
        out = self.fully_connect(out)
        return out

# </editor-fold>



# <editor-fold desc="**4** Training & Testing loop">
def saveTrack(target_label, output_label,
              target_pro, output_pro,
              epoch, num_epochs, epoch_loss):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    f1 = f1_score(target_label, output_label, average='binary', zero_division=0)
    cm = confusion_matrix(target_label, output_label).reshape(1, 4)[0]
    TN, FP, FN, TP = cm[0], cm[1], cm[2], cm[3]
    FPR, FNR = FP / (FP + TN), FN / (FN + TP)


    logLossActual = log_loss(target_label, target_pro)
    logLossPredicted = log_loss(target_label, output_pro)
    brierScoreLoss = brier_score_loss(target_label, output_pro)

    f = open(f"{OUTPUT_DIR}{OUTPUT_NAME}_epochOUTPUT.txt", 'a')
    if epoch == "Testing":
        epoch, num_epochs = 0, 1
        record = f"{now}, Testing[{epoch}/{num_epochs}], Loss,{epoch_loss:.4e}, " \
                 f"F1,{f1:.4f}, FPR,{FPR}, FNR,{FNR}, TN,{TN}, FP,{FP}, FN,{FN}, TP,{TP}, " \
                 f"logLossActual,{logLossActual:.3e}, " \
                 f"logLossPredicted,{logLossPredicted:.3e}, " \
                 f"brierScoreLoss,{brierScoreLoss:.3e}"
    else:
        record = f"{now}, Training[{epoch}/{num_epochs}], Loss,{epoch_loss:.4e}, " \
                 f"F1,{f1:.4f}, FPR,{FPR}, FNR,{FNR}, TN,{TN}, FP,{FP}, FN,{FN}, TP,{TP}, " \
                 f"logLossActual,{logLossActual:.3e}, " \
                 f"logLossPredicted,{logLossPredicted:.3e}, " \
                 f"brierScoreLoss,{brierScoreLoss:.3e}"

    f.write(str(record) + "\n")
    f.close()


def visualizeConfusionMatrix(target_label, output_label, training_or_testing):

    cm_raw = confusion_matrix(target_label, output_label)
    cm_df_raw = pd.DataFrame(cm_raw, index=["0:None DF", "1:DF"], columns=["0:None DF", "1:DF"])

    cm_normalize = confusion_matrix(target_label, output_label, normalize='true')
    cm_df_normalize = pd.DataFrame(cm_normalize, index=["0:None DF", "1:DF"], columns=["0:None DF", "1:DF"])

    f1 = f1_score(target_label, output_label, average='binary', zero_division=0)

    fig = plt.figure(figsize=(4.5, 4.5))

    sns.heatmap(cm_df_raw, xticklabels=1, yticklabels=1, annot=True, fmt='.0f', square=True, cmap="Blues", cbar=False)

    plt.text(x=0.35, y=0.62, s=f"{cm_df_normalize.iloc[0, 0]:.4f}", color="black")
    plt.text(x=1.35, y=0.62, s=f"{cm_df_normalize.iloc[0, 1]:.4f}", color="black")

    plt.text(x=0.35, y=1.62, s=f"{cm_df_normalize.iloc[1, 0]:.4f}", color="black")
    plt.text(x=1.35, y=1.62, s=f"{cm_df_normalize.iloc[1, 1]:.4f}", color="black")

    plt.ylabel("Actual Class", weight='bold')
    plt.xlabel(f"Predicted Class" + "\n" + f"{training_or_testing}, {STATION}, F1={f1:.4}", weight='bold')

    plt.tight_layout()
    if training_or_testing == "Training":
        plt.savefig(f"{OUTPUT_DIR}{STATION}_LSTM_s{sequenceLEN}b{batchSIZE}_trainingF1.png", dpi=600)
    elif training_or_testing == "Testing":
        plt.savefig(f"{OUTPUT_DIR}{STATION}_LSTM_s{sequenceLEN}b{batchSIZE}_testingF1.png", dpi=600)
    plt.close(fig)

    return f1


def saveAllOutput(targetLabelList, outputLabelList,
                   targetProList, outputProList,
                   timeList, train_or_test):
    datetime_objects = pd.to_datetime(timeList, unit='s', utc=True)
    utc0Time = datetime_objects.strftime('%Y-%m-%d %H:%M:%S').values

    outputProList = np.round(outputProList, decimals=3) # save as 3 decimal digits
    # Combine arrays into a 2D NumPy array
    targetPro0List = 1 - targetProList
    data = np.column_stack((utc0Time, timeList, targetLabelList, targetPro0List, targetProList, outputLabelList, outputProList))
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]

    if train_or_test == "train":
        path = f"{OUTPUT_DIR}{OUTPUT_NAME}_{train_or_test}Out2017-2019.txt"
    else:
        path = f"{OUTPUT_DIR}{OUTPUT_NAME}_{train_or_test}Out{utc0Time[0][:4]}.txt"
    np.savetxt(path, sorted_data, delimiter=',', fmt='%s',
               header='time, timeStamps, label, Pro0, Pro1, PredictedLabel, PredictedProbability', comments='')


def validationModel(featureSIZE, dataloader):
    test_model = LstmClassifier(feature_size=featureSIZE, dropout=0.25)
    load_ckp = f"{OUTPUT_DIR}{OUTPUT_NAME}_checkPoint.pt"
    test_model.load_state_dict(torch.load(load_ckp, map_location=torch.device('cpu')))
    test_model.to(DEVICE)

    tester = LSTM_Tester(test_model, dataloader)
    tester.testing_step()

class LSTM_Trainer:
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_dataloader: DataLoader):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader

        # give second output_logit more weight [0.01noise, 0.99DF]
        class_weight = torch.tensor([0.1, 0.9], dtype=torch.float64).to(DEVICE)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean", weight=class_weight)

    def save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        path = f"{OUTPUT_DIR}{OUTPUT_NAME}_checkPoint.pt"

        torch.save(ckp, path)
        print(f"Model saved at epoch {epoch}, {path}", flush=True)


    def training_step(self):
        min_loss, num_epoch = 1, 50 # min_loss will be updated
        self.model.train()

        for epoch in range(num_epoch): # loop 50 times for training
            targetLabelList, outputLabelList = [], []
            targetProList, outputProList = [], []
            timeList = []

            for batch_data in self.train_dataloader:
                input = batch_data['features'].to(DEVICE)  # Shape: (batch_size, sequence_length, input_size)
                target = batch_data['label'].to(DEVICE)    # Shape: (sequence_length) 0:NoneDF, 1:DF

                output_logit = self.model(input)  # return the model output logits
                loss = self.loss_func(output_logit, target)

                # update the gredient
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # add the outputs and targets to list
                target_pro = batch_data['pro'].cpu().numpy().astype('float64')
                target_label = target.cpu().numpy().astype('float64')

                output_pro = F.sigmoid(output_logit)[:, 1].cpu().detach().numpy().astype('float64') # pro of label 1 DF
                output_label = (output_pro > 0.5).astype('float64') # pro bigger than 0.5 as label 1 DF

                time = batch_data['timestamps'].cpu().numpy().astype('float64')

                # add the elements into list
                targetLabelList.extend(list(target_label))# target label
                outputLabelList.extend(list(output_label))# model output label

                targetProList.extend(list(target_pro))# target Probability
                outputProList.extend(list(output_pro))# model output  Probability

                timeList.extend(list(time))

            epoch_loss = loss.item()
            # Move the lists to the CPU after the training loop
            targetProList = np.array(targetProList)
            targetLabelList = np.array(targetLabelList)

            outputProList = np.array(outputProList)
            outputLabelList = np.array(outputLabelList)
            timeList = np.array(timeList)

            # save training track
            saveTrack(target_label=targetLabelList, output_label=outputLabelList,
                      target_pro=targetProList, output_pro=outputProList,
                      epoch=epoch, num_epochs=num_epoch, epoch_loss=epoch_loss)

            # save the model if loss decrease
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                self.save_checkpoint(epoch)

                visualizeConfusionMatrix(target_label=targetLabelList,
                                         output_label=outputLabelList,
                                         training_or_testing="Training")

                saveAllOutput(targetLabelList, outputLabelList,
                              targetProList, outputProList,
                              timeList, "train")


class LSTM_Tester:
    def __init__(self, model: torch.nn.Module,
                 test_dataloader: DataLoader):
        self.model = model
        self.test_dataloader = test_dataloader

        class_weight = torch.tensor([0.1, 0.9], dtype=torch.float64).to(DEVICE)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean", weight=class_weight)

    def testing_step(self):
        self.model.eval()

        targetLabelList, outputLabelList = [], []
        targetProList, outputProList = [], []
        timeList = []

        # loop all testing data
        with torch.no_grad():
            for batch_data in self.test_dataloader:
                input = batch_data['features'].to(DEVICE)  # Shape: (batch_size, sequence_length, input_size)
                target = batch_data['label'].to(DEVICE)  # Shape: (sequence_length) 0:NoneDF, 1:DF

                output_logit = self.model(input)  # .softmax(dim=1) #return the probability of [ProNonEvent0, ProEvent1]
                loss = self.loss_func(output_logit, target)

                # add the outputs and targets to list
                target_pro = batch_data['pro'].cpu().numpy().astype('float64')
                target_label = target.cpu().numpy().astype('float64')

                output_pro = F.sigmoid(output_logit)[:, 1].cpu().detach().numpy().astype('float64')  # pro of label 1 DF
                output_label = (output_pro > 0.5).astype('float64')  # pro bigger than 0.5 as label 1 DF

                time = batch_data['timestamps'].cpu().numpy().astype('float64')

                # add the elements into list
                targetLabelList.extend(list(target_label))  # target label
                outputLabelList.extend(list(output_label))  # model output label

                targetProList.extend(list(target_pro))  # target Probability
                outputProList.extend(list(output_pro))  # model output  Probability

                timeList.extend(list(time))

        epoch_loss = loss.item()
        # Move the lists to the CPU after the training loop
        targetProList = np.array(targetProList)
        targetLabelList = np.array(targetLabelList)

        outputProList = np.array(outputProList)
        outputLabelList = np.array(outputLabelList)
        timeList = np.array(timeList)

        # save training track
        saveTrack(target_label=targetLabelList, output_label=outputLabelList,
                  target_pro=targetProList, output_pro=outputProList,
                  epoch="Testing", num_epochs=1, epoch_loss=epoch_loss)

        visualizeConfusionMatrix(target_label=targetLabelList, output_label=outputLabelList,
                                 training_or_testing="Testing")

        saveAllOutput(targetLabelList, outputLabelList,
                      targetProList, outputProList,
                      timeList, "test")


class LSTM_TT:
    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # give second output_logit more weight [0.01noise, 0.99DF]
        class_weight = torch.tensor([0.1, 0.9], dtype=torch.float64).to(DEVICE)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean", weight=class_weight)

        self.min_loss = 1
        self.testF1 = 0

    def save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        path = f"{OUTPUT_DIR}{OUTPUT_NAME}_checkPoint.pt"

        torch.save(ckp, path)
        print(f"Model saved at epoch {epoch}, {path}", flush=True)

    def training(self, epoch, num_epoch):
        self.model.train()

        targetLabelList, outputLabelList = [], []
        targetProList, outputProList = [], []
        timeList = []

        for batch_data in self.train_dataloader:
            input = batch_data['features'].to(DEVICE)  # Shape: (batch_size, sequence_length, input_size)
            target = batch_data['label'].to(DEVICE)  # Shape: (sequence_length) 0:NoneDF, 1:DF

            output_logit = self.model(input)  # return the model output logits
            loss = self.loss_func(output_logit, target)

            # update the gredient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # add the outputs and targets to list
            target_pro = batch_data['pro'].cpu().numpy().astype('float64')
            target_label = target.cpu().numpy().astype('float64')

            output_pro = F.sigmoid(output_logit)[:, 1].cpu().detach().numpy().astype('float64')  # pro of label 1 DF
            output_label = (output_pro > 0.5).astype('float64')  # pro bigger than 0.5 as label 1 DF

            time = batch_data['timestamps'].cpu().numpy().astype('float64')

            # add the elements into list
            targetLabelList.extend(list(target_label))  # target label
            outputLabelList.extend(list(output_label))  # model output label

            targetProList.extend(list(target_pro))  # target Probability
            outputProList.extend(list(output_pro))  # model output  Probability

            timeList.extend(list(time))

        epoch_loss = loss.item()
        # Move the lists to the CPU after the training loop
        targetProList = np.array(targetProList)
        targetLabelList = np.array(targetLabelList)

        outputProList = np.array(outputProList)
        outputLabelList = np.array(outputLabelList)
        timeList = np.array(timeList)

        # save training track
        saveTrack(target_label=targetLabelList, output_label=outputLabelList,
                  target_pro=targetProList, output_pro=outputProList,
                  epoch=epoch, num_epochs=num_epoch, epoch_loss=epoch_loss)

        # save the model if loss decrease
        if epoch_loss < self.min_loss:
            self.min_loss = epoch_loss
            self.save_checkpoint(epoch)

            visualizeConfusionMatrix(target_label=targetLabelList,
                                     output_label=outputLabelList,
                                     training_or_testing="Training")

            saveAllOutput(targetLabelList, outputLabelList,
                          targetProList, outputProList,
                          timeList, "train")

    def testing(self, epoch):
        self.model.eval()

        targetLabelList, outputLabelList = [], []
        targetProList, outputProList = [], []
        timeList = []

        # loop all testing data
        with torch.no_grad():
            for batch_data in self.test_dataloader:
                input = batch_data['features'].to(DEVICE)  # Shape: (batch_size, sequence_length, input_size)
                target = batch_data['label'].to(DEVICE)  # Shape: (sequence_length) 0:NoneDF, 1:DF

                output_logit = self.model(input)  # .softmax(dim=1) #return the probability of [ProNonEvent0, ProEvent1]
                loss = self.loss_func(output_logit, target)

                # add the outputs and targets to list
                target_pro = batch_data['pro'].cpu().numpy().astype('float64')
                target_label = target.cpu().numpy().astype('float64')

                output_pro = F.sigmoid(output_logit)[:, 1].cpu().detach().numpy().astype('float64')  # pro of label 1 DF
                output_label = (output_pro > 0.5).astype('float64')  # pro bigger than 0.5 as label 1 DF

                time = batch_data['timestamps'].cpu().numpy().astype('float64')

                # add the elements into list
                targetLabelList.extend(list(target_label))  # target label
                outputLabelList.extend(list(output_label))  # model output label

                targetProList.extend(list(target_pro))  # target Probability
                outputProList.extend(list(output_pro))  # model output  Probability

                timeList.extend(list(time))

        epoch_loss = loss.item()
        # Move the lists to the CPU after the training loop
        targetProList = np.array(targetProList)
        targetLabelList = np.array(targetLabelList)

        outputProList = np.array(outputProList)
        outputLabelList = np.array(outputLabelList)
        timeList = np.array(timeList)


        f1 = visualizeConfusionMatrix(target_label=targetLabelList, output_label=outputLabelList,
                                 training_or_testing="Testing")

        # save training track
        saveTrack(target_label=targetLabelList, output_label=outputLabelList,
                  target_pro=targetProList, output_pro=outputProList,
                  epoch="Testing", num_epochs=epoch, epoch_loss=epoch_loss)


        if f1 > self.testF1: # save the best test model
            self.testF1 = f1
            ckp = self.model.state_dict()
            path = f"{OUTPUT_DIR}{OUTPUT_NAME}_checkPoint_test.pt"
            torch.save(ckp, path)
            print(f"Test Model saved at epoch {epoch}, {path}", flush=True)

            saveAllOutput(targetLabelList, outputLabelList,
                          targetProList, outputProList,
                          timeList, "test")

    def activation(self, num_epoch=50):

        for epoch in range(1, num_epoch+1): # loop 50 times for training
            self.training(epoch, num_epoch) # train the model every epoch

            if epoch % 5 == 0: # test the model every 5 epoch
                self.testing(epoch)

# </editor-fold>


# <editor-fold desc="**5** run calculate loop">
def run(featureSIZE, scalerData):
    scaler = MinMaxScaler()

    # <editor-fold desc="load data as df">
    if featureTYPE == "C":
        df = load_C_Features(STATION)
    elif featureTYPE == "A" or featureTYPE == "B":
        df = load_A_or_B_Features(STATION, featureTYPE)
    elif featureTYPE == "D":
        df = load_D_Features(STATION)

    train_df, test_df = splitDataFrame(df, splitID = 509760)
    print("finished DF load", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n", flush=True)

    if scalerData == True:
        scaler.fit(train_df.iloc[:, :-3])
        train_df.iloc[:, :-3] = scaler.transform(train_df.iloc[:, :-3])
        test_df.iloc[:, :-3] = scaler.transform(test_df.iloc[:, :-3])
    # </editor-fold>


    # <editor-fold desc="convert train and test df as dataloader">
    train_sequences = DataToSeq(df=train_df, seq_length=sequenceLEN)
    train_dataloader = DatasetToDataloader(data_sequences=train_sequences, batch_size=batchSIZE, training_or_testing="training")

    test_sequences = DataToSeq(df=test_df, seq_length=sequenceLEN)
    test_dataloader = DatasetToDataloader(data_sequences=test_sequences, batch_size=batchSIZE, training_or_testing="testing")

    print("finished dataloader", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")
    # </editor-fold>


    # <editor-fold desc="2017-2019 train, 2020 test data">
    train_model = LstmClassifier(feature_size=featureSIZE, dropout=0.25)
    train_model.to(DEVICE)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)

    # train the model
    trainer = LSTM_TT(train_model, optimizer,
                      train_dataloader.dataLoader(),
                      test_dataloader.dataLoader())
    trainer.activation()
    print("finished training", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    # </editor-fold>


    # <editor-fold desc="Re-load 2020 test data">
    test_model = LstmClassifier(feature_size=featureSIZE, dropout=0.25)
    load_ckp = f"{OUTPUT_DIR}{OUTPUT_NAME}_checkPoint_test.pt"
    test_model.load_state_dict(torch.load(load_ckp, map_location=torch.device('cpu')))
    test_model.to(DEVICE)

    # use the 2020 data retest the model
    tester2020 = LSTM_Tester(test_model, test_dataloader.dataLoader())
    tester2020.testing_step()

    # </editor-fold>
# </editor-fold>


# <editor-fold desc="**6** define main loop">
def main(input_station:str, feature_type:str, sequence_length:int, batch_size:int):

    global STATION, featureTYPE, sequenceLEN, batchSIZE, featureSIZE,\
        DATA_DIR, OUTPUT_DIR, OUTPUT_NAME, DEVICE

    STATION = input_station
    featureTYPE = feature_type
    sequenceLEN = sequence_length
    batchSIZE = batch_size

    DATA_DIR = f"/home/qizhou/1projects/dataForML/out60finish/processed/"
    OUTPUT_DIR = f"/home/qizhou/3paper/2AGU/2LSTM/LSTM_{featureTYPE}/"
    OUTPUT_NAME = f"{STATION}_LSTM_s{sequenceLEN}b{batchSIZE}_{featureTYPE}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Start Job: {STATION}, {featureTYPE}, {sequenceLEN}, {batchSIZE}, {DEVICE}: ",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")

    if featureTYPE == "A":
        featureSIZE = 11 + 2
    elif featureTYPE == "B":
        featureSIZE = 69
    elif featureTYPE == "C":
        featureSIZE = 80
    elif featureTYPE == "D":
        featureSIZE = 6

    run(featureSIZE, scalerData=True)

    print(f"End Job: {STATION}, {featureTYPE}, {sequenceLEN}, {batchSIZE}, {DEVICE}: ",
          datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# </editor-fold>



if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres" # check the available GPU on glic
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--input_station", default="ILL12", type=str, help="data source")
    parser.add_argument("--feature_type", default="rf", type=str, help="feature type")
    parser.add_argument("--sequence_length", default=64, type=int, help="Input sequence length")
    parser.add_argument("--batch_size", default=16, type=int, help='Input batch size on each device')

    args = parser.parse_args()
    main(args.input_station, args.feature_type, args.sequence_length, args.batch_size)


#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import numpy as np
from scipy.signal import hilbert

from datetime import datetime, timedelta
from obspy import read, Trace
from obspy.core import UTCDateTime # default is UTC+0 time zone

import matplotlib.pyplot as plt

def create_trace(data, low_sampling_rate, ref_st):
    '''
    create Obspy st
    Args:
        data: numpy 1D data array, unit by m/s or other
        low_sampling_rate: int or float, unit by Hz
        ref_st: obspy st, suppose the st1 = read(), here ref_st = st1[0]

    Returns:
        created Obspy st, as ref_st structure

    '''
    trace = Trace(data=data)
    trace.stats.sampling_rate = low_sampling_rate

    # get the ref information
    trace.stats.network = ref_st.stats.network
    trace.stats.station = ref_st.stats.station
    trace.stats.starttime = ref_st.stats.starttime
    trace.stats.channel = ref_st.stats.channel

    st = Stream([trace])

    return st

def envelop(signal):
    '''
    calculate the envelop of a seismic signal

    Args:
       signal: 1D numpy array, time series seismic signal

    Returns:
        amplitude_envelope: 1D numpy array,

    '''

    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)

    return amplitude_envelope

def denoising_method(chunk_x, denoising_processing_method, row_or_column):
    '''
    provide different denoising methods

    Args:
        chunk_x: 2D numpy array, each row represents one time step
        denoising_processing_method: str, denoising methods
        row_or_column: str, 0 denotes to column, 1 denotes row

    Returns:
        x_value: 1D numpy array,
        x_value.shape[0] should equal to chunk_x.shape[0]

    '''
    if row_or_column == "row":
        row_or_column = 1
    elif row_or_column == "column":
        row_or_column = 0
    else:
        print(f"check the row_or_column = {row_or_column}")

    if denoising_processing_method == "RMS":
        x_value = np.sqrt(np.mean(chunk_x ** 2, axis=row_or_column))
    elif denoising_processing_method == "IQR":
        x_q75 = np.percentile(chunk_x, 75, axis=row_or_column)
        x_q25 = np.percentile(chunk_x, 25, axis=row_or_column)
        x_value = x_q75 - x_q25
    else:
        print(f"check the denoising_processing_method = {denoising_processing_method}")

    return x_value

def seismic_denoising(seismic_data, data_start, seismic_sampling_rate, x_window_size, window_ovelap, denoising_processing_method):
    '''
    receive high_sampling_rate seismic data and use "RMS" or "IQR" to denosise the data in low_sampling_rate

    Args:
        seismic_data: 1D numpy array, unit by m/s
        data_start: str, start time of the seismic data, no physical unit
        seismic_sampling_rate: int, seismic data sampling rate, unit by Hz
        x_window_size: int, window size for praparing x data, unit by second
        window_ovelap: float, overlap ratio for each time step, 0->no overlap, 0.75->every step get 1/4 new data, no physical unit
        denoising_processing_method: str, either "RMS" or "IQR", no physical unit

    Returns:
        t_value, x_value: 1D numpy array, smae as input physical unit
        low_sampling_rate (unit by Hz) = 1 / (x_window_size * (1 - window_ovelap))
    '''

    x_seq_length = int(seismic_sampling_rate * x_window_size)  # unit by data point
    overlap_length = int(x_seq_length * (1 - window_ovelap))  # unit by data point

    # prepare the float time stamps
    start_datetime = datetime.strptime(data_start, "%Y-%m-%dT%H:%M:%S")
    timestamps = np.array([start_datetime + timedelta(seconds= i / seismic_sampling_rate) for i in range(len(seismic_data))])
    timestamps = np.array([(ts - datetime(1970, 1, 1)).total_seconds() for ts in timestamps])
    # you can use this to convert the float time back to string
    # datetime.utcfromtimestamp(t_value[1]).strftime('%Y-%m-%d %H:%M:%S.%f')

    # prepare a 2D data numpy array
    if window_ovelap != 0:  # for overlap window
        chunk_t = np.lib.stride_tricks.sliding_window_view(timestamps, x_seq_length)[::overlap_length]
        chunk_x = np.lib.stride_tricks.sliding_window_view(seismic_data, x_seq_length)[::overlap_length]
    else:
        num_windows = len(seismic_data) // x_seq_length
        chunk_t = timestamps[:num_windows * x_seq_length].reshape(-1, x_seq_length)
        chunk_x = seismic_data[:num_windows * x_seq_length].reshape(-1, x_seq_length)

    t_value = chunk_t[:, 0]
    x_value = denoising_method(chunk_x, denoising_processing_method, "row")
    low_sampling_rate = 1 / (x_window_size * (1 - window_ovelap))

    return t_value, x_value, low_sampling_rate


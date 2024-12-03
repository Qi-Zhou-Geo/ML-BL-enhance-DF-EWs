#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-11-02
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission


import os
import platform
import sys

import argparse

from datetime import datetime

import pytz

import numpy as np
import pandas as pd

from obspy import read, Stream, read_inventory, signal
from obspy.core import UTCDateTime # default is UTC+0 time zone


# <editor-fold desc="define the parent directory">
if platform.system() == 'Darwin':
    parent_dir = "/Users/qizhou/#python/#GitHub_saved/ML-BL-enhance-DF-EWs"
elif platform.system() == 'Linux':
    parent_dir = "/home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs"
else:
    print(f"check the parent_dir for platform.system() == {platform.system()}")
# add the parent_dir to the sys
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
else:
    pass
# </editor-fold>


# import the custom functions
from config.config_dir import CONFIG_dir, path_mapping
from seismic_data_processing import load_seismic_signal # load and process the seismic signals
from archive_data import save_hdf5
from hilbert_envelop import *


def check_folder(seismic_network, input_year, input_station, input_component):
    '''

    Args:
        seismic_network:
        input_year:
        input_station:
        input_component:

    Returns:

    '''

    folder_path = f"{CONFIG_dir['feature_output_dir']}/{path_mapping(seismic_network)}/{input_year}/{input_station}/{input_component}"
    CONFIG_dir['txt_dir'] = folder_path
    os.makedirs(f"{CONFIG_dir['txt_dir']}", exist_ok=True)


def denoise(tr, f_min, f_max, x_window_size=30, window_ovelap=0, denoising_processing_method="IQR"):
    '''

    Args:
        tr:
        f_min:
        f_max:
        x_window_size: unit is second
        window_ovelap:
        denoising_processing_method:

    Returns:

    '''

    # filter the trace
    tr.filter("bandpass", freqmin=f_min, freqmax=f_max)
    tr.detrend('linear')
    tr.detrend('demean')

    # convert to envelop and denosing the data
    seismic_envelop = envelop(tr.data)
    data_start = tr.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")
    seismic_sampling_rate = tr.stats.sampling_rate

    t_value, x_value, low_sampling_rate = seismic_denoising(seismic_envelop,
                                                            data_start,
                                                            seismic_sampling_rate,
                                                            x_window_size,
                                                            window_ovelap,
                                                            denoising_processing_method)
    # save as 2D array
    file_dir = CONFIG_dir['txt_dir']
    file_name = f"{tr.stats.network}-{tr.stats.station}-{tr.stats.channel}-" \
                f"{tr.stats.starttime.strftime('%Y')}-" \
                f"{str(tr.stats.starttime.strftime('%j')).zfill(3)}"

    dataset = np.column_stack((t_value, x_value))
    dataset_name = f"{file_name}-{f_min}-{f_max}"
    metadata = {
        "description": "deconvolved amplitude (2D array)",

        "data_sampling_rate": low_sampling_rate,
        "data_start_time_float": tr.stats.starttime.timestamp,
        "data_start_time_string": tr.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S.%j"),
        "data_time_zone": "UTC+0",
        "data_unit": "meter per second",
        "f_min": f_min,
        "f_max": f_max,

        "data_creation_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "author": "Qi Zhou"
    }

    save_hdf5(file_dir, file_name, dataset_name, dataset, metadata)

    try:
        lock_path = f"{file_dir}/{file_name}.lock"
        os.remove(lock_path)
    except FileNotFoundError:
        pass



def cal_loop(seismic_network, input_year, julday_id, input_station, input_component, length=24):
    '''

    Args:
        seismic_network:
        input_year:
        julday_id:
        input_station:
        input_component:
        length: unit is hour

    Returns:

    '''

    data_start = UTCDateTime(year=input_year, julday=julday_id)
    data_end = data_start + length * 3600

    st = load_seismic_signal(seismic_network, input_station, input_component, data_start, data_end,
                             f_min=1, f_max=45, remove_sensor_response=True)
    st = st[0]
    st.data = st.data * 1e9 # convert from m/s to nm/s

    f1 = [1, 5,  10, 15, 20, 25, 30, 35] # lower frequency boundary
    f2 = [5, 10, 15, 20, 25, 30, 35, 40] # upper frequency boundary
    for step in np.arange(len(f1)):
        tr = st.copy()
        denoise(tr, f1[step], f2[step])



def main(seismic_network, input_year, input_station, input_component, input_window_size, id):
    '''

    Args:
        seismic_network: str, seismic network
        input_year: int, data year
        input_station:
        input_component:
        input_window_size:
        id:

    Returns:

    '''

    job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"Start Job {job_id}: {input_year}, {input_station}: ", datetime.now().strftime("%Y-%m-%dT%H:%M:%S") )

    # check the folder
    try:
        check_folder(seismic_network, input_year, input_station, input_component)
    except Exception as e:
        print(f"{seismic_network, input_year, input_station, input_component, input_window_size, id}, \n"
              f"Exception {e}")

    if seismic_network in ["CC"]: # data segment
        cal_loop_seg(seismic_network, input_year, input_station, input_component, input_window_size, id1, id2)
    else:
        cal_loop(seismic_network, input_year, id, input_station, input_component, length=24)

    print(f"End Job {job_id}: {input_year}, {input_station}: ", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seismic_network", type=str, default="9S", help="check the year")
    parser.add_argument("--input_year", type=int, default=2020, help="check the year")
    parser.add_argument("--input_station", type=str, default="ILL12", help="check the input_station")
    parser.add_argument("--input_component", type=str, default="EHZ", help="check the input_component")
    parser.add_argument("--input_window_size", type=int, default=60, help="check the calculate window size")
    parser.add_argument("--id", type=int, default=60, help="check the julday id")

    args = parser.parse_args()
    main(args.seismic_network, args.input_year, args.input_station, args.input_component, args.input_window_size, args.id)

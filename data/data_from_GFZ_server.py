#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-03-21
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do NOT distribute this code without the author's permission

import platform
import numpy as np

from obspy import read, Stream, UTCDateTime, read_inventory


global SAC_PATH

if platform.system() == "Darwin": # your local PC name
    SAC_PATH = "/Volumes/Section_4.7/0data/"
elif platform.system() == "Linux":  # your remote server name
    SAC_PATH = "/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/CN/"

        

def load_seismic_signal(data_start, data_end, station, component="EHZ", remove_sensor_response=False):
    '''
    Load seismic signal

    Parameters:
    - data_start (str): the start time to select data, e.g., 2017-04-03 12:00:00
    - data_end (str): the start time to select data, e.g., 2017-04-03 13:00:00
    - station (str): seismic station name
    - component (str): seismic component name
    - remove_sensor_response (bool, optial): for deconvolove

    Returns:
    - st (obspy.core.stream): seismic stream
    '''

    d1 = UTCDateTime(data_start)
    d2 = UTCDateTime(data_end)

    sac_dir = f"{SAC_PATH}{d1.year}/{station}/{component}/"

    if d1.year in [2013, 2014]:
        seismic_network = "GM"
    elif d1.year in [2017, 2018, 2019, 2020]:
        seismic_network = "9S"


    if d1.julday == d2.julday:
        data_name = f"{seismic_network}.{station}.{component}.{d1.year}.{str(d1.julday).zfill(3)}"
        st = read(sac_dir + data_name)
    else:
        st = Stream()
        for n in np.arange(d1.julday, d2.julday+1):
            data_name = f"{seismic_network}.{station}.{component}.{d1.year}.{str(n).zfill(3)}"
            st += read(sac_dir + data_name)


    st = st.trim(starttime=d1, endtime=d2, nearest_sample=False)
    st.merge(method=1, fill_value='latest', interpolation_samples=0)
    st._cleanup()
    st.detrend('linear')
    st.detrend('demean')
    st.filter("bandpass", freqmin=1, freqmax=45)


    if remove_sensor_response is True:
        inv = read_inventory(f"{SAC_PATH}metadata_2017-2020.xml")
        st.remove_response(inventory=inv)

    return st


def remove_sensor_response(trace, sensorType):
    '''
    visuzlice the PSD

    Parameters:
    - st (obspy.core.stream): seismic stream that deconvolved, make sure the stream only hase one trace
    - sensorType (str): sensor type

    Returns:
    - st (obspy.core.stream): seismic stream that removed the sensor response
    '''

    from obspy.signal.invsim import simulate_seismometer
    corrected_trace = trace.copy()

    paz_geophone = {
        'poles': [(-15.88 + 23.43j),
                  (-15.88 - 23.43j)],
        'zeros': [0j, 0j],
        'gain': 28.8,
        'sensitivity': 1.3115e8
    }

    paz_trillum = {
        'poles': [(-0.037004 + 0.037016j),
                  (-0.037004 - 0.037016j),
                  (-251.33 + 0j),
                  (-131.04 - 467.29j),
                  (-131.04 + 467.29j)],
        'zeros': [0j, 0j],
        'gain': 60077000.0,
        'sensitivity': 629145000.0
    }

    paz_SmartSolo_IGU_16_5hz = {
        'poles': [-22.211059 + 22.217768j, -22.211059 + 22.217768j],
        'zeros': [0j, 0j],
        'gain': 24,
        'sensitivity': 7.68e1}

    if sensorType == "geophone":
        sensorLogger = paz_geophone
    elif sensorType == "trillum":
        sensorLogger = paz_trillum
    elif sensorType == "SmartSolo_5Hz":
        sensorLogger = paz_SmartSolo_IGU_16_5hz
    else:
        print("please check the sensorType")

    corrected_data = simulate_seismometer(trace[0].data, trace[0].stats.sampling_rate,
                                          paz_remove=sensorLogger,  # or paz_trillum depending on your choice
                                          remove_sensitivity=True)


    corrected_trace[0].data = corrected_data

    return corrected_trace


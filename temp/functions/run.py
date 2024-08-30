import os
import sys
import argparse
from typing import List
import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream

from utils import _get_juldays, _get_data_duration, _get_user_request_df, _get_miniseed_data
from plot import make_subplot
from calculate_features import BL_loop

sys.path.append("../../calculate_features/")
from seismic_data_processing import load_seismic_signal

def run_cal_loop(st:Stream, year:int, ruler:int, juldays:List) -> pd.DataFrame:
    BL_dataframe = None
    print("Running Benford's Law feature calculation loop")
    for day in sorted(juldays):
        print(f"Calculating for day {day}")
        bl_df = BL_loop(st= st, output_dir="NA", year=year, component="EHZ", station="ILL12",
                    julday = int(day), ruler= ruler, save_cal=False)
        if BL_dataframe is None:
            BL_dataframe = bl_df
        else:
            BL_dataframe = pd.concat([BL_dataframe, bl_df], axis=0).reset_index(drop=True)
    print("Done calculating Benford's Law features.")
    return BL_dataframe


def main(ruler:int, scaling_factor:int, index:int= None, data_start:UTCDateTime= None, data_end:UTCDateTime= None) -> None:
    if index is not None:
        df = _get_user_request_df(index=index)
        juldays = list(_get_juldays(df=df))
        year = UTCDateTime(df['Start(UTC+0)']).year
        x = UTCDateTime(year= year, julday = int(juldays[0]))
        y = UTCDateTime(year= year, julday = int(juldays[-1])+1)
        print(f"Data Start time : {x}\n Data End Time : {y}")
        network, station, component = _get_miniseed_data(year= year)
        print("Loading seismic data")
        st = load_seismic_signal(seismic_network= network, station= station,
                                component= component, data_start= x, data_end= y)
        st[0].data = st[0].data * (10**scaling_factor)
        print("Loaded seismic data")
        BL_dataframe = run_cal_loop(st= st, year= year, ruler= ruler, juldays= juldays)
        data_duration, x_interval = _get_data_duration(start_time= x, end_time= y)
        make_subplot(st= st, ruler= ruler, outdir=".", DataStart= x, DF_df= df, 
                     DataDuration= data_duration, x_interval= x_interval, BL_data= BL_dataframe)
        print("Done")
    
    if data_start is not None and data_end is not None:
        x = UTCDateTime(data_start)
        y = UTCDateTime(data_end)
        print(f"Data Start time : {x}\n Data End Time : {y}")
        year = x.year
        juldays = _get_juldays(start_time= x, end_time= y)
        network, station, component = _get_miniseed_data(year= year)
        print("Loading seismic data")
        st = load_seismic_signal(seismic_network= network, station= station,
                                component= component, data_start= x, data_end= y)
        st[0].data = st[0].data * (10**scaling_factor)
        print("Loaded seismic data")
        BL_dataframe = run_cal_loop(st= st, year= year, ruler= ruler, juldays= juldays)
        data_duration, x_interval = _get_data_duration(start_time= x, end_time= y)
        make_subplot(st= st, ruler= ruler, outdir=".", DataStart= x, DF_df= df, 
                     DataDuration= data_duration, x_interval= x_interval, BL_data= BL_dataframe)
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler", type= int, default= 1e2, help= "input ruler")
    parser.add_argument("--scaling", type= int, default= 9, help= "input the scaling factor. eg: 9 for 1e9")
    parser.add_argument("--index", type= int, default= None, help= "Input index of DF from list of DFs")
    parser.add_argument("--data_start", type= str, default= None, help= "Input start time in yyyy-mm-dd hh:mm:ss format")
    parser.add_argument("--data_end", type= str, default= None, help= "Input start time in yyyy-mm-dd hh:mm:ss format")

    args = parser.parse_args()
    main(ruler= args.ruler, scaling_factor= args.scaling, index= args.index, data_start= args.data_start, data_end= args.data_end)



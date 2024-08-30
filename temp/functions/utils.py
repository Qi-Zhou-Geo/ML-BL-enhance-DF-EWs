from typing import List, Tuple
import pandas as pd
from obspy import UTCDateTime


def _get_user_request_df(index:int=None, start_time:UTCDateTime=None, end_time:UTCDateTime=None) -> pd.DataFrame | pd.Series:
    if index is not None:
        df = pd.read_csv("../DF_events/58 DF-BL events.txt", header=0, index_col="Index")
        if index < len(df):
            df = df.iloc[index]
            return df
        else:
            print(f"Index {index} is greater than the number of DF events ({len(df)})")
    else:
        pass

    if start_time is not None and end_time is not None:
        df = pd.read_csv("../DF_events/58 DF-BL events.txt", header=0, index_col="Index")
        df = df[df["Start(UTC+0)"] <= end_time]
        df = df[df[" End(UTC+0)"] >= start_time]
        return df
    else:
        pass

def _get_juldays(df:pd.DataFrame | pd.Series = None, start_time:UTCDateTime = None, end_time:UTCDateTime = None) -> List:
    julday_set = set()
    if df is not None:
        if type(df) == pd.Series:
            julday_set.add(str(UTCDateTime(df['Start(UTC+0)']).julday).zfill(3))
            julday_set.add(str(UTCDateTime(df[' End(UTC+0)']).julday).zfill(3))
        else:
            print(f"Enter a pandas Series object. Entered {type(df)}")
    
    if start_time is not None and end_time is not None:
        for i in range(start_time.julday, end_time.julday+1):
            julday_set.add(str(i).zfill(3))

    return list(julday_set)

def _get_data_duration(start_time:UTCDateTime, end_time:UTCDateTime) -> Tuple[float, float]:
    return (end_time - start_time) // 3600, (end_time - start_time) // (3600 * 5)

def _get_miniseed_data(year:int) -> Tuple[str, str, str]:
    if year in [2013,2014]:
        component = "HHZ"
        station = "IGB02"
        network = "9J"
    elif year in [2017]:
        component = "EHZ"
        station = "ILL02"
        network = "9S"
    elif year in [2018, 2019, 2020]:
        component = "EHZ"
        station = "ILL12"
        network = "9S"
    return network, station, component
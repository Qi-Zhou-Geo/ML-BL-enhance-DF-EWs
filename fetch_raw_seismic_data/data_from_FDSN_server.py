#!/usr/bin/python
# -*- coding: UTF-8 -*-


#__modification time__ = 2024-02-04
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import argparse
import platform
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# <editor-fold desc="**0** set input-output path">
def set_outputPath(year):
    '''
    Parameters
    ----------
    year: year of request data; integer
    -------
    Returns: no return
    -------
    '''
    global OUTPUT_DIR

    OUTPUT_DIR = "check-the-path"
    if platform.system() == "Darwin":  # your local PC name
        OUTPUT_DIR = f"/Volumes/Section_4.7/1seismic_data/{year}/"
    elif platform.system() == "Linux":  # your remote server name
        OUTPUT_DIR = f"/home/qizhou/#data/{year}/"
    elif platform.system() == "Windows":  # your local PC name
        OUTPUT_DIR = f"H:/1seismic_data/{year}/"

    return OUTPUT_DIR

# </editor-fold>2LSTM_glic_danu_commented

# <editor-fold desc="**1** downloader">

def download_seismicData(clientName, julian_day, year, network, station, component):
    '''
    Parameters
    ----------
    clientName: sever/client name in FDSN; string
    julian_day: julian day; float or string
    year: year of request data; integer
    network: seismic network code in FDSN; string
    station: station name; string
    component: component name; string
    -------
    Returns: no return
    -------
    '''
    client = Client(clientName)

    julian_day = str(julian_day).zfill(3)
    start = UTCDateTime(year=year, julday=julian_day, hour=00, minute=00)
    end = start + 86400 # 86400 seconds = 1 day

    output_mseed_dir = f"{OUTPUT_DIR}{station}/{component}/"
    output_mseed_fileName = f"{network}.{station}.{component}.{year}.{julian_day}"

    try:
        st = client.get_waveforms(network=network, station=station, location="--",
                                  channel=component, starttime=start, endtime=end)
        st.write(filename=f"{output_mseed_dir}{output_mseed_fileName}.mseed", format="MSEED")
        status = "success"
    except Exception as e:
        print(f"{output_mseed_fileName}, {e}")
        status = "failure"

    f = open(f"{OUTPUT_DIR}log.txt", 'a')
    f.write(f"{output_mseed_fileName}, {status}" + "\n")
    f.close()

# </editor-fold>2LSTM_glic_danu_commented


def main(clientName, julian_day, year, network, station, component):
    download_seismicData(clientName, julian_day, year, network, station, component)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='set the input data')

    parser.add_argument("--clientName", default="GFZ", type=str, help="please pass the right client name")
    parser.add_argument("--julian_day", default="001", type=str, help="julian day")
    parser.add_argument("--network", default="9S", type=str, help="seismic network code in FDSN")
    parser.add_argument("--year", default="2013", type=str, help="seismic network code in FDSN")
    parser.add_argument("--station", default="IGB01", type=str, help="seismic station name")
    parser.add_argument("--component", default="EHZ", type=str, help="seismc channel or companet")

    args = parser.parse_args()

    main(args.clientName, args.julian_day, args.network, args.year, args.station, args.component)


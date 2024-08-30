from __future__ import division, print_function
import pytz
import numpy as np
import pandas as pd
from datetime import datetime
from obspy import Stream
from obspy.core import UTCDateTime # default is UTC+0 time zone
from scipy.stats import iqr, ks_2samp, chi2_contingency, mannwhitneyu
from scipy.signal import hilbert, lfilter, butter, spectrogram
from scipy.stats import kurtosis, skew

##################################################################################
# BL FEATURES
##################################################################################

def BL_loop(st:Stream, output_dir:str, year:int, component:str, station:str,
            julday:int, ruler:int= 1e2, KS_thr:float= 0.95, MW_thr:float= 0.95, save_cal:bool= True) -> pd.DataFrame:
    stationMapping = {'ILL11': 1.0, 'ILL12': 2.0, 'ILL13': 3.0,
                      'ILL14': 4.0, 'ILL15': 5.0, 'ILL16': 6.0,
                      'ILL17': 7.0, 'ILL18': 8.0}
    componentMapping = {'EHZ': 1.0, 'EHE': 2.0, 'EHN': 3.0, 'BHZ': 1.0, 'BHE': 2.0, 'BHN': 3.0}
    input_window_size = 60
    d = UTCDateTime(year=year, julday= julday)
    output = None
    for step in range(0, int((3600*24)/input_window_size)):
        d1 = d + step * input_window_size
        d2 = d + (step + 1) * input_window_size
        timeDate = datetime.fromtimestamp(d2.timestamp, tz=pytz.utc)
        timeDate = timeDate.strftime('%Y-%m-%d %H:%M:%S')
        id = np.array([timeDate, d2.timestamp, 
                        componentMapping.get(component, -1),
                        stationMapping.get(station, -1)]).reshape(1,-1)
        tr = st.copy()
        try:
            tr.trim(starttime=d1, endtime=d2, nearest_sample=False)
            # CALCULATE BL FEATURES
            output_bl = calBL_feature(data= tr[0].data, ruler= ruler)
        except IndexError:
            output_bl = np.full((1,17), np.nan)
        output_bl = np.append(id, output_bl, axis= 1)
        if output is None:
            output = output_bl
        else:
            output = np.append(output, output_bl, axis= 0)

    feature_names = ['timeDate', 'time_stamps', 'station', 'component',
                     'digit1','digit2','digit3','digit4','digit5',
                     'digit6','digit7','digit8','digit9',
                     'max', 'min', 'iq', 'goodness', 'alpha', 'ks', 'MannWhitneU', 'followOrNot']
    feature_names = np.array(feature_names)

    pd_output = pd.DataFrame(output, columns= feature_names)
    if save_cal:
        pd_output.to_csv(f"{output_dir}/{julday}_BL.csv")
        return None
    return pd_output
        

def calBL_feature(data:np.array, ruler:int= 1e1, KS_thr:float = 0.95, MW_thr:float = 0.95) -> np.array:
    '''
    Parameters
    ----------
    
    - data (np.array): np array data after detrend, demean, filter, deconv, and amplitude as envelope
    - ruler (int): add the "small" data if it less than "ruler"
    
    Returns: pd.DataFrame
    -------
    '''
    # BL theoretical value
    BL_frequency = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    none_value_alpha = 10

    data_abs = np.abs(data)
    dataSelected = data_abs[data_abs >= ruler]

    # <editor-fold desc="iq, max, min">
    iqr_value = float("{:.2f}".format(iqr(data_abs)))
    max_amp = float("{:.2f}".format(np.max(data_abs)))
    min_amp = float("{:.2f}".format(np.min(data_abs)))
    # </editor-fold>

    if len(dataSelected) == 0:
        goodness, alpha, ks, MannWhitneU, follow = -150, none_value_alpha, 0, 0, 0
        digit_frequency = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        # <editor-fold desc="digit frequency">
        amp_data = pd.DataFrame(dataSelected)
        amp_data = amp_data.astype(str)

        d = (amp_data.iloc[:, 0]).str[0: 1]
        d = list(d)

        digit_count = np.empty((0, 9))
        for digit in range(1, 10):
            first_digit = d.count(str(digit))
            digit_count = np.append(digit_count, first_digit)

        digit_frequency = digit_count / np.sum(digit_count)
        digit_frequency = [float('{:.3f}'.format(i)) for i in digit_frequency]
        # </editor-fold>

        # <editor-fold desc="get goodness, ks, chi-squared, alpha">
        frequency = np.empty((0, 9))
        for a in range(0, 9):
            first_digit_frequency = pow((digit_frequency[a] - BL_frequency[a]), 2) / BL_frequency[a]
            frequency = np.append(frequency, first_digit_frequency)
        goodness = (1 - pow(sum(frequency), 0.5)) * 100
        goodness = float("{:.3f}".format(goodness))

        statistic, pvalue = ks_2samp(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
        ks = float("{:.3f}".format(pvalue))  # pvalue

        statistic, pvalue = mannwhitneyu(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
        MannWhitneU = float("{:.3f}".format(pvalue))  # pvalue

        follow = 1 if ks >= KS_thr and MannWhitneU >= MW_thr else 0

        sum_d = []
        y_min = np.nanmin(dataSelected)
        y_min = 1 if y_min == 0 else y_min

        for s in range(0, len(dataSelected)):
            i = np.log(dataSelected[s] / y_min)
            sum_d.append(i)

        if np.sum(sum_d) != 0:    
            alpha = 1 + len(dataSelected) / np.sum(sum_d)
        else:
            alpha = 0
        alpha = float("{:.4f}".format(alpha))
        # </editor-fold>

    output = np.array([max_amp, min_amp, iqr_value, goodness, alpha, ks, MannWhitneU, follow], dtype=float)
    output = np.append(digit_frequency, output)
    output = output.reshape(1,-1)

    return output
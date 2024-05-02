#__author__ = "Qi Zhou, https://github.com/Nedasd"

import numpy as np
import pandas as pd
from scipy.stats import iqr, ks_2samp, chi2_contingency, mannwhitneyu


# the BL is based on the following refrence
# Sambridge, Malcolm, Hrvoje Tkalčić, and A. Jackson. "Benford's law in the natural sciences." Geophysical research letters 37.22 (2010).
# Zhou, Qi, et al. "Benford's law as mass movement detector in seismic signals." (2023).
def calBL_feature(data, ruler=1e1):
    '''
    Parameters
    ----------
    data: np array data after dtrend+dmean+filter 1-45 Hz, or raw data
    ruler: discard the "small" data if it less than "ruler"
    Returns: np.array
    -------
    '''
    data = np.abs(data)
    #hilb = hilbert(data)  # envolope
    #data = (data ** 2 + hilb ** 2) ** 0.5
    # BL theoretical value
    BL_frequency = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    dataSelected = data[data >= ruler]

    # <editor-fold desc="iq, max, min">
    iq = float("{:.2f}".format(iqr(dataSelected)))
    max_amp = float("{:.2f}".format(np.max(dataSelected)))
    min_amp = float("{:.2f}".format(np.min(dataSelected)))
    # </editor-fold>

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

    # chi squared methods not works well
    # digit_frequency = [x / sum(digit_frequency) for x in digit_frequency] # for chi squared
    #statistic, pvalue = chisquare(f_obs=digit_frequency, f_exp=BL_frequency)
    #chi = float("{:.3f}".format(pvalue)) # pvalue

    statistic, pvalue = mannwhitneyu(BL_frequency, digit_frequency, alternative='two-sided', method='exact')
    MannWhitneU = float("{:.3f}".format(pvalue))  # pvalue

    if ks >= 0.95 and MannWhitneU >= 0.95:
        follow = 1 # follow BL
    else:
        follow = 0 # do not follow BL

    sum_d = []
    y_min = np.nanmin(dataSelected)
    if y_min == 0: # in csse of "divide by zero encountered in scalar divide"
        y_min = 1
    else:
        pass

    for s in range(0, len(dataSelected)):
        i = np.log(dataSelected[s] / y_min)
        sum_d.append(i)
    alpha = 1 + len(dataSelected) / np.sum(sum_d)
    alpha = float("{:.4f}".format(alpha))
    # </editor-fold>

    output = np.array([max_amp, min_amp, iq, goodness, alpha, ks, MannWhitneU, follow], dtype=float)
    output = np.append(digit_frequency, output)
    return output


# the FD is based on the following refrence
# Genton, Marc G. "The correlation structure of Matheron's classical variogram estimator under elliptically contoured distributions." Mathematical geology 32 (2000): 127-137.
# Tosi, Patrizia, et al. "Seismic signal detection by fractal dimension analysis." Bulletin of the Seismological Society of America 89.4 (1999): 970-977.
def calFD_feature(data, SPS, num=4):
    '''
    Parameters
    ----------
    data: data after dtrend+dmean, or raw data, np array
    SPS: sampling frequency of the seismic data, int
    num: how many data to select to fit the double-log curve

    Returns: Fractal diminasion, float
    -------
    '''

    # to store timeLag, tau, gamma_tau
    raw_array = np.empty((0, 3))
    # how many timeLag could be gerenated
    # because we only use 4 points to fit the curve, for minuing the calculation time,
    # here we only use 50
    max_timeLag = 50#len(data)-1

    # <editor-fold desc=" calculate the ">

    # timeLag = 1 is 1/SPS unit in real word
    for timeLag in range(1, max_timeLag):
        # make sure that each id in seq1 minus seq2 equals to timeLag
        sequence1 = data[ timeLag: ]
        sequence2 = data[ 0: len(data)-timeLag ]

        versusDistance = sequence1 - sequence2
        versusDistance = np.sum(versusDistance * versusDistance)

        Nh = len(sequence1) # len(sequence1) +  timeLag = max_timeLag
        tau = timeLag / SPS
        gamma_tau = versusDistance/ (2 * Nh)

        record = np.array([timeLag, tau, gamma_tau], dtype=float)
        raw_array = np.vstack((raw_array, record))
    # </editor-fold>


    # <editor-fold desc="fit the slope in double-log coordinates">
    x = raw_array[0:num, 1]
    x = np.log10(x)

    y = raw_array[0:num, 2]
    y = np.log10(y)

    z = np.polyfit(x, y, deg=1)

    FD = 2 - z[0] / 2
    if FD > 2:
        FD = 2

    FD = round(FD, 3)
    # </editor-fold>

    return FD

#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-02-23
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import numpy as np
import pandas as pd
from scipy.stats import iqr, ks_2samp, chi2_contingency, mannwhitneyu


# the BL is based on the following refrence
# Sambridge, Malcolm, Hrvoje Tkalčić, and A. Jackson. "Benford's law in the natural sciences." Geophysical research letters 37.22 (2010).
# Zhou, Qi, et al. "Benford's law as mass movement detector in seismic signals." (2023).
def calBL_feature(data, ruler):
    '''
    Parameters
    ----------
    data: np array data after dtrend+dmean+filter 1-45 Hz, or raw data
    ruler: discard the "small" data if it less than "ruler"
    Returns: np.array
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

        if ks >= 0.95 and MannWhitneU >= 0.95:
            follow = 1  # follow BL
        else:
            follow = 0  # do not follow BL

        sum_d = []
        y_min = np.nanmin(dataSelected)
        if y_min == 0:  # in csse of "divide by zero encountered in scalar divide"
            y_min = 1
        else:
            pass

        for s in range(0, len(dataSelected)):
            i = np.log(dataSelected[s] / y_min)
            sum_d.append(i)

        if np.sum(sum_d) == 0:
            alpha = none_value_alpha
        else:
            alpha = 1 + len(dataSelected) / np.sum(sum_d)
            alpha = float("{:.4f}".format(alpha))
        # </editor-fold>

    output = np.array([max_amp, min_amp, iqr_value, goodness, alpha, ks, MannWhitneU, follow], dtype=float)
    output = np.concatenate((digit_frequency, output), axis=0)

    return output

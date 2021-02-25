# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:35:19 2020

@author: admin
"""

import pandas as pd
import scipy.signal as scisig
import os
import numpy as np
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

def peak_detection(df_EDA):
    f = 4  # サンプリング周波数
    threshold = 0.01  # 閾値（振幅がこれより大きいものをSCRとして検出）

    df_EDA['filtered_eda'] =  butter_lowpass_filter(df_EDA['EDA'], 1.0, f, order=5)
    EDA_shift = df_EDA['filtered_eda'][1:].values - df_EDA['filtered_eda'][:-1].values

    peaks = np.zeros(len(EDA_shift))
    peak_sign = np.sign(EDA_shift)
    bottoms = np.zeros(len(EDA_shift))
    peak_starts = np.zeros(len(EDA_shift))

    for i in range(len(EDA_shift)-1):
        if peak_sign[i] == -1 and peak_sign[i+1] == 1:
            bottoms[i+1] = 1
        if peak_sign[i] == 1 and peak_sign[i+1] == -1:
            peaks[i+1] = 1

    peak_locs = np.where(peaks == 1)
    bottom_locs = np.where(bottoms == 1)
    df_peak = pd.Series(peak_locs[0], name='Peak')
    df_bottom = pd.Series(bottom_locs[0], name='Bottom')

    if df_peak[0] < df_bottom[0]:
        df_peak = df_peak[1:].reset_index(drop=True)
    if df_peak[len(df_peak)-1] < df_bottom[len(df_bottom)-1]:
        df_bottom = df_bottom[:-1].reset_index(drop=True)

    PeakInfo = pd.concat([df_peak,df_bottom], axis=1)
    PeakInfo['PeakStart'] = PeakInfo['Bottom']

    for i in range(len(PeakInfo)-1):
        if i == 0:
            pass
        else:
            if PeakInfo['Peak'][i] - PeakInfo['Peak'][i-1] < 4:
                if df_EDA['filtered_eda'][PeakInfo['Bottom'][i]] >= df_EDA['filtered_eda'][PeakInfo['PeakStart'][i-1]] :
                    PeakInfo['PeakStart'][i] = PeakInfo['PeakStart'][i-1]
                else:
                    pass

    PeakInfo['PeakValue'] = df_EDA['filtered_eda'][PeakInfo['Peak']].reset_index(drop=True)
    PeakInfo['PeakStartValue'] = df_EDA['filtered_eda'][PeakInfo['PeakStart']].reset_index(drop=True)
    PeakInfo['Amplitude'] = PeakInfo['PeakValue'] - PeakInfo['PeakStartValue']

    SCR_param = pd.DataFrame()
    SCR_Param = PeakInfo[ PeakInfo['Amplitude'] > threshold ].reset_index(drop=True)
    SCR_Param['RiseTime'] = (SCR_Param['Peak'] - SCR_Param['PeakStart']) / f
    SCR_Param['HalfRecoveryTime'] = 0

    half_times = []
    HalfRecovery_window = 100  # 1/2回復時間を探すときのウィンドウ

    for i in range(len(SCR_Param)):
        peak_loc = SCR_Param['Peak'][i]
        half_loc = peak_loc
        half_amplitude = SCR_Param['Amplitude'][i] * 0.5
        found = 0
        while half_loc < half_loc + HalfRecovery_window and found == 0 and half_loc < len(df_EDA):
            if half_amplitude <= df_EDA['filtered_eda'][peak_loc] -df_EDA['filtered_eda'][half_loc]:
                # SCR_Param['HalfRecoveryTime'][i] =  (half_loc - peak_loc) / f
                half_times = np.append(half_times, (half_loc - peak_loc) / f)
                found = 1

            half_loc += 1
        if found == 0:
            # SCR_Param['HalfRecoveryTime'][i] = HalfRecovery_window
            half_times = np.append(half_times, 0)

    SCR_Param['HalfRecoveryTime'] = half_times

    SCR_Param.rename(columns={'Peak': 'PeakTime', 'Bottom': 'BottomTime', 'PeakStart': 'PeakStartTime'}, inplace=True)

    SCR_Param['PeakTime'] = SCR_Param['PeakTime'] / f
    SCR_Param['BottomTime'] = SCR_Param['BottomTime'] / f
    SCR_Param['PeakStartTime'] = SCR_Param['PeakStartTime'] / f

    return df_EDA, SCR_Param

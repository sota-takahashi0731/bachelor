import numpy as np
import pandas as pd

def time_sync(data, t_unix, t_start, t_end):
    ts = 0
    #データのスタート時刻を合わせる
    for i, ti in enumerate(t_unix):
        if ts==0 and int(ti)>=int(t_start):
            ts = i
        if int(ti)>=t_end:
            eda_data = data[ts:i]
            t_data = t_unix[ts:i]
            break
    print(ts, i)
    t_data = [(t_data[i]-t_data[0]) for i in range(len(t_data))]
    return eda_data, t_data

def ibi_data(df_ibi):
    t_ibi = df_ibi.values[:, 0]
    ibi = df_ibi.values[:, 1]
    t_unix_ibi = [(t_ibi[i]+float(df_ibi.columns[0])) for i in range(t_ibi.shape[0])]
    ibi = ibi*1000
    return ibi, t_unix_ibi

def bvp_data(df_bvp):
    bvp = df_bvp.values[1:, 0]
    t_unix_bvp = [(0.015625*i+float(df_bvp.columns[0])) for i in range(bvp.shape[0])]
    return bvp, t_unix_bvp

def eda_data(df_eda):
    eda = df_eda.values[1:, 0]
    t_unix_eda = [(0.25*i+float(df_eda.columns[0])) for i in range(eda.shape[0])]
    return eda, t_unix_eda

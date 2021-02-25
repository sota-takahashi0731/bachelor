import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import numpy.linalg as LA
from statsmodels.tsa import ar_model
import scipy as sp
import datetime as dt
import scipy.stats as stats
from scipy import integrate


###SG検定による異常値除去
def smirnov_grubbs(data, t, alpha):
    x, o = list(data), []
    while True:
        n = len(x)
        t1 = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
        tau = (n - 1) * t1 / np.sqrt(n * (n - 2) + n * t1 * t1)
        i_min, i_max = np.argmin(x), np.argmax(x)
        myu, std = np.mean(x), np.std(x, ddof=1)
        i_far = i_max if np.abs(x[i_max] - myu) > np.abs(x[i_min] - myu) else i_min
        tau_far = np.abs((x[i_far] - myu) / std)
        if tau_far < tau: break
        o.append(i_far)
        x.pop(i_far)
    for i in o:
        t = np.delete(t, i)
    return np.array(x), np.array(t)


def outlier_rm(data, t):
    #心拍変動データの三次スプライン補完
    t = np.array(t)

    dataf = sp.interpolate.interp1d(t,data,kind="cubic")
    t = np.array(range(int(t.max())))

    data = dataf(t)
    #異常値除去+欠損値補間
    data, t= smirnov_grubbs(data, t, 0.05)
    dataf = sp.interpolate.interp1d(t, data, kind="linear")
    data = dataf(t)

    return data, t

def RRI_data(df_rri, date):
    df_rri['time'] = date + df_rri['time']
    rri = df_rri.values[:, 1]
    t_datetime_rri = [dt.datetime.strptime(df_rri['time'][i], '%Y/%m/%d %H:%M:%S') for i in range(df_rri.shape[0])]
    t_unix_rri = [0] * df_rri.shape[0]
    t_unix_rri[0] = t_datetime_rri[0].timestamp()
    for i in range(1, df_rri.shape[0]):
        t_unix_rri[i] = t_unix_rri[i-1]+rri[i]*0.001
    return rri, t_unix_rri

###PSDの計算
def psd_cal(data):
    model = ar_model.AR(data)
    results = model.fit(maxlag=12)
    k = 12
    N = 512
    Y = results.predict()
    A = results.params    #係数ai
    res = results.resid   #分散σ^2
    s = res.std(ddof=0)
    s2 = s**2


    # Frequency = np.fft.fftfreq(N, d=dt)
    freq = np.arange(0, 0.5, 0.5/N)
    P= []


    # for f in Frequency[:int(N/2)]:
    for f in freq:
        S = 0
        for i in range(1, results.k_ar + 1):
            S += results.params[i] * (math.e)**(-2 * math.pi * 1j * i * f)
        psd = s2 / (abs(1 - S))**2
        P = np.append(P, psd)
    return P, freq



###LP面積の計算
def LP_cal(data):
    N = data.shape[0]
    #データ可視化
    RRIn = np.array(data[:N-1])
    RRIn1 = np.array(data[1:])
    a = np.column_stack((RRIn, RRIn1))
    N = a.shape[0]

    ###LP面積算出
    #y=xへの投影
    pro_yx = np.zeros((N, 2))
    b = np.array([1,1])
    for i in range(N):
        pro_yx[i, :] = np.dot(a[i,:], b)/np.dot(b, b) *b
    #y=-xへの投影
    pro_y_x = np.zeros((N, 2))
    b = np.array([1,-1])
    for i in range(N):
        pro_y_x[i,:] = np.dot(a[i,:], b)/np.dot(b, b) * b
    #ノルムの計算
    d_yx = np.zeros(N)
    d_y_x = np.zeros(N)
    for i in range(N):
        d_yx[i] = LA.norm(pro_yx[i, :], ord=2)
        d_y_x[i] = LA.norm(pro_y_x[i, :], ord=2)
    #標準偏差の算出
    s_yx = np.std(d_yx)
    s_y_x = np.std(d_y_x)
    S = math.pi * s_yx * s_y_x
    d = np.mean(d_yx)
    return S, s_yx/s_y_x, np.log10(s_yx*s_y_x)

###LF,HFの計算
def linerequation(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b

def Integrate(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    y = lambda x: a*x + b
    S, err = integrate.quad(y, x1, x2)
    return S

def calc_LF_HF(Pow, freq):
    # LF, HFの閾値の設定
    VLF_lower, VLF_upper = 0, 0.05
    LF_lower, LF_upper = 0.05, 0.15
    HF_lower, HF_upper = 0.15, 0.40
    VLF = 0
    # for i in range(int(N/2) + 1):
    for i in range(len(freq)):
        if VLF_lower <= freq[i] < VLF_upper:
            if freq[i-1] <= VLF_lower:
                a1, b1 = linerequation(freq[i-1], Pow[i-1], freq[i], Pow[i])
                y1 = lambda x: a1 * x + b1
                VLF, err = integrate.quad(y1, VLF_lower, freq[i])

                vlf = Integrate(freq[i], Pow[i], freq[i+1], Pow[i+1])
                VLF += vlf

            elif VLF_upper <= freq[i+1]:
                a1, b1 = linerequation(freq[i], Pow[i], freq[i+1], Pow[i+1])
                y1 = lambda x: a1 * x + b1
                vlf, err = integrate.quad(y1, freq[i], VLF_upper)
                VLF += vlf

            else:
                vlf = Integrate(freq[i], Pow[i], freq[i+1], Pow[i+1])
                VLF += vlf

        elif LF_lower < freq[i] < LF_upper:
            if freq[i-1] <= LF_lower:
                a1, b1 = linerequation(freq[i-1], Pow[i-1], freq[i], Pow[i])
                y1 = lambda x: a1 * x + b1
                LF, err = integrate.quad(y1, LF_lower, freq[i])

                lf = Integrate(freq[i], Pow[i], freq[i+1], Pow[i+1])
                LF += lf

            elif LF_upper <= freq[i+1]:
                a1, b1 = linerequation(freq[i], Pow[i], freq[i+1], Pow[i+1])
                y1 = lambda x: a1 * x + b1
                lf, err = integrate.quad(y1, freq[i], LF_upper)
                LF += lf

            else:
                lf = Integrate(freq[i], Pow[i], freq[i+1], Pow[i+1])
                LF += lf

        elif  0.15 <= freq[i] < 0.40:
            if freq[i-1] <= HF_lower:
                a1, b1 = linerequation(freq[i-1], Pow[i-1], freq[i], Pow[i])
                y1 = lambda x: a1 * x + b1
                HF, err = integrate.quad(y1, HF_lower, freq[i])

                hf = Integrate(freq[i], Pow[i], freq[i+1], Pow[i+1])
                HF += hf

            elif HF_upper <= freq[i+1]:
                a1, b1 = linerequation(freq[i], Pow[i], freq[i+1], Pow[i+1])
                y1 = lambda x: a1 * x + b1
                hf, err = integrate.quad(y1, freq[i], HF_upper)
                HF += hf

            else:
                hf = Integrate(freq[i], Pow[i], freq[i+1], Pow[i+1])
                HF += hf

    return VLF, LF, HF

def HR_cal(RRI):
    RRI = np.array(RRI)
    HR = np.empty_like(RRI)
    for i, rri in enumerate(RRI):
        HR[i] = 60000/rri
    return HR

def params_cal(RRI):
    RRI = np.array(RRI)
    N = RRI.shape[0]

    ##時系列指標
    mean = np.mean(RRI)  #平均値
    SDNN = np.std(RRI)  #標準偏差
    #pNN50の計算
    pNNcnt = 0
    diff = [0]*(N-1)
    for i in range(N-1):
        diff[i] = RRI[i+1]-RRI[i]
        if abs(diff[i]) > 50:
            pNNcnt+=1
    pNN50 = pNNcnt/(N-1)
    #RMSSDの計算
    RMSSD = 0
    for i in range(N-1):
        RMSSD += diff[i]**2
    RMSSD = math.sqrt(RMSSD/(N-1))

    ##PSD関連パラメータ
    P, freq = psd_cal(RRI)
    VLF, LF, HF = calc_LF_HF(P, freq)
    TotalPow = VLF + LF + HF

    ###幾何学指標
    S, CSI = LP_cal(RRI)

    params = [mean, SDNN, pNN50, RMSSD, VLF, LF, HF, LF/HF, TotalPow, S]

    return params

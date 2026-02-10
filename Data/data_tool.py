import torch
import torch.nn as nn
import os
import pywt
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

class moving_avg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size 
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        front = x[0:1, :].repeat((self.kernel_size - 1)//2, 1)
        end = x[-1:, :].repeat(self.kernel_size //2, 1)
        x = torch.cat([front, x, end], dim=0)
        x = self.avg(x.permute(1, 0))
        x = x.permute(1, 0)
        return x

class decomp(nn.Module):

    def __init__(self, kernel_size):
        super(decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        cycle, trend = sm.tsa.filters.hpfilter(moving_mean)
        res = x - moving_mean
        return cycle, trend, res

def series_decomp(data, kernel_size=25):
    data_ = np.array(data)
    data_ = torch.from_numpy(data_)
    decomp_ = decomp(kernel_size)
    cycle, trend, res = decomp_(data_)
    return cycle, trend, res

def get_cycle(cycle, topk):
    cycle = torch.tensor(cycle)
    freq = torch.fft.fft(cycle)
    freq = torch.abs(freq)
    seq_len = cycle.shape[0]
    freq = torch.topk(freq[1: int(seq_len/2)], k = topk, dim=0).indices + 1 
    season = seq_len / freq
    seasonal_inf = torch.round(season)
    return seasonal_inf

def seasonal_interpolation(data, index):
    for col in data.columns:
        get_data = data[col]
        df = pd.DataFrame(get_data.tolist(), index=index)
        interpolated_data = df.interpolate(method='time', axis=0)
        data_ = interpolated_data[0]
        length = len(get_data)
        seasonal_inf = get_cycle(data_, 10).tolist()        
        for i in range(10):
            if seasonal_inf[i] < length / 10:
                seasonal = seasonal_inf[i]
                break
        index_ = range(0, length)
        df = pd.DataFrame(get_data, index=index_)
        missing_indices = df[df[col].isnull()]
        for i in missing_indices.index:
            k = i - i // seasonal*seasonal
            x = []
            y = []
            while(k < length):
                if k not in missing_indices.index:
                    x.append(k)
                    y.append(get_data[k])
                k += seasonal
            model_trend = LinearRegression()
            x = np.array(x).reshape(-1, 1)
            y = np.array(y).reshape(-1, 1)
            model_trend.fit(x, y)
            target = np.array(i).reshape(-1, 1)
            get_data[i] = model_trend.predict(target)
        data[col] = get_data
    return data     

def pywt_decomp(data, layer, type):
    wavelet = type  # Daubechies 4 
    level = layer   
    coeffs = pywt.wavedec(data, wavelet, level=level)
    trend = pywt.waverec(np.multiply(coeffs, [1] + [0] * level).tolist(), wavelet)
    seasonal = data - trend
    return trend, seasonal

def trend_extract(data):
    B, T, N = data.shape
    window = 25
    trend_ma = np.zeros((B, T, N))
    pad_width = window // 2
    data = data.cpu()
    for b in range(B):
        for n in range(N):
            x = data[b, :, n]
            x_padded = np.pad(x, pad_width, mode='reflect')
            ma = np.convolve(x_padded, np.ones(window)/window, mode='valid')        
            trend_ma[b, :, n] = ma

    trend_ma = torch.from_numpy(trend_ma).float().to(data.device)
    return trend_ma
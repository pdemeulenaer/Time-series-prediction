import pandas as pd
import numpy as np
from pandas import datetime
import matplotlib.pyplot as plt
from datetime import datetime
import calendar

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import signal

try:
    import statsmodels.api as sm
except:
    print('No statsmodel package installed')

import random


def time_series_generator(size=635,
                          cycle_period=30.5,
                          signal_type='sine',
                          salary=1,
                          trend=0.1,
                          noise=0.1,
                          offset=0,
                          spike=0):
    '''
    Synthetic time series generator
    Input : 
    Output: - ts : the generated time series
    '''
    
    #x = np.linspace(-0.5*30.5*21, 0.5*30.5*21, 635)
    #phase_random = np.random.uniform(-1,1)
    #phi = np.pi
    #s1 = np.sin(2*np.pi*(1./30.5)*x + 0  ) + 2
    #s2 = np.sin(2*np.pi*(1./30.5)*x +phase_random*phi) + 2    
    
    # size : length of the time series
    # cycle_period : period of the signal (usually 30.5, the month period, in days)
    # count_periods : number of periods in the time series
    # in size = 635, and cycle_period = 30.5, we have ~ 21 periods (20.8)
    count_periods = size / cycle_period
    
    # 1. The trend making
    t = np.linspace(-0.5*cycle_period*count_periods, 0.5*cycle_period*count_periods, size)
    t_trend = np.linspace(0, 1, size)
    trend = trend*salary*t_trend**2          
  
    # 2. The seasonality making
    if offset == 1: 
        phase = np.random.uniform(-1,1)*np.pi
    else: 
        phase = 0           
        
    if signal_type == 'mixed'   : 
        choice = np.random.randint(4, size=1)
        if choice == 0 : signal_type = 'sine'
        if choice == 1 : signal_type = 'sawtooth'
        if choice == 2 : signal_type = 'triangle'
        if choice == 3 : signal_type = 'square'
    if signal_type == 'sine':     ts = 0.25*salary*np.sin(2*np.pi*(1./cycle_period)*t + phase)    
    if signal_type == 'sawtooth': ts = -0.25*salary*signal.sawtooth(2*np.pi*(1./cycle_period)*t + phase)
    if signal_type == 'triangle': ts = 0.5*salary*np.abs(signal.sawtooth(2*np.pi*(1./cycle_period)*t + phase))-1
    if signal_type == 'square':   ts = 0.25*salary*signal.square(2*np.pi*(1./cycle_period)*t + phase)
           
    # 3. The noise making
    noise = np.random.normal(0,noise*salary,size)  
            
    ts = ts + trend + noise
            
    # 4. Adding spikes to the time series
    if spike > 0: 
        for spike_i in range(spike):
            sign = random.choice([-1,1])
            t_spike = np.random.randint(0,455) #size)
            ts[t_spike:] = ts[t_spike:] + sign * np.random.normal(3*salary,salary)
                
    return ts


def scale_data(X,y,N_samples,N_days_X,N_days_y,scale_type=2):
    # Fit only to the training data # ONE SCALER PER TIME SERIES!!! 
    scalers_list = []
    X_scaled = np.zeros((N_samples,N_days_X))
    if y is not None: y_scaled = np.zeros((N_samples,N_days_y))
    for ts in range(0,N_samples):
        if scale_type==1: scaler = MinMaxScaler()
        if scale_type==2: scaler = StandardScaler()
        if scale_type==3: scaler = RobustScaler()
        scaler.fit(X[ts,:].reshape(-1, 1)) #the reshaping is because scikit-learn deals with 2D arrays
        X_scaled[ts] = scaler.transform(X[ts,:].reshape(-1, 1)).reshape(N_days_X)
        if y is not None: y_scaled[ts] = scaler.transform(y[ts,:].reshape(-1, 1)).reshape(N_days_y)    
        scalers_list.append(scaler)
        del scaler
       
    #Reshaping data (no: leave that out of the function: better to keep same dimension as input)
    #X_scaled = X_scaled.reshape(N_samples,N_days_X,1)
    #y_scaled = y_scaled.reshape(N_samples,N_days_y,1)                
    if y is not None: 
        return X_scaled, y_scaled, scalers_list
    else:
        return X_scaled, None, scalers_list


def scale_data_back(data_scaled,scalers_list):
    '''
    This function scales back a signal that has been scaled (with MinMaxScaler,StandardScaler or RobustScaler)
    Note that here data_scaled could be X or y.
    Input:
           - data_scaled  : data (2D numpy array) scaled
           - scalers_list : list of scikit-learn scalers. List dimension = data_scaled.shape[0]
    '''
    # Scaling BACK prediction to normal scale # ONE SCALER PER TIME SERIES!!!
    
    #Reshaping data
    # (no: leave that out of the function: better to keep same dimension as input)
    #y_pred_reshaped_diff_scaled = y_pred.reshape(N_samples_pred,N_days_y)
    
    N_samples = data_scaled.shape[0]
    N_days = data_scaled.shape[1]

    data_scaled_back = np.zeros((N_samples,N_days))
    for ts in range(0,N_samples):
        data_scaled_back[ts,:] = scalers_list[ts].inverse_transform(data_scaled[ts])

    return data_scaled_back


def extrapolate_trend(trend_array,slope_array,N_days_y):    
    '''
    This function extrapolates the long-term trend of the time series. It takes the
    median of the slope of the last 6 months trend
    
    Input:  - trend_array       : Numpy array containing the trends of the time series
            - slope_array       : Numpy array containing the slopes of trends of the time series
            - N_days_y          : the length of the prediction time series (usually 3 months)
           
    Output: - trend_reconstruct : Numpy array containing the extrapolation of the trend over 
                                  N_days_y time steps
    '''
    
    slope_fixed = np.median(slope_array.values[-183:]) #
    
    #print(slope_fixed,trend_array.values[-1])
    
    trend_reconstruct = np.zeros(N_days_y)
    trend_reconstruct[0] = trend_array.values[-1]
    for xx in range(1,N_days_y):
        trend_reconstruct[xx] = slope_fixed*1 + trend_reconstruct[xx-1]    
    
    return trend_reconstruct
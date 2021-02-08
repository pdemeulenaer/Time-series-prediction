# Databricks notebook source
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
from pandas import Series
import numpy as np
from datetime import datetime, date
import time as tm
import logging
import argparse
import traceback
import json
import matplotlib.pyplot as plt
import random
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
#from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf,PandasUDFType
from pyspark.sql.types import ArrayType, FloatType
# import tensorflow as tf
# from mlflow.tracking.client import MlflowClient


# =====================
# FUNCTIONS
# =====================

#   Example of usage in main file:
#   >>from utils import cli_arguments, setup_logger
#   >>parser = cli_arguments()
#   #Add your own arguments
#   >>parser.add_argument('--subset', help='Use a subset of data', default=False, action='store_true')
#   >>args = parser.parse_args()
#   >>hosting_env = args.env
#   >>debug = args.debug
#   >>subset = args.subset
#   >>if debug:
#   >>    logger = setup_logger('my_app', level=logging.DEBUG)
#   >>else:
#   >>    logger = setup_logger('my_app')
#   >>logger.info('This will always be logged')

def cli_arguments():
    '''
    Helper function to set up an argument parser.
    :return: Argument parser object
    '''
    parser = argparse.ArgumentParser(description='Command-line arguments')
    parser.add_argument('--env', choices=['ddp', 'odl'], help='Hosting environment', required=True)
    parser.add_argument('--debug', help='Log at debug level', default=False, action='store_true')

    return parser


def Get_Data_From_JSON(file):
    '''
    This function extracts a configuration json file as a python dictionary
    :param (string) file: the name of the configuration file
    :return (dict) json_dict: the dictionary containing the configuration parameters
    '''
    r_value = 0

    try:
        with open(file, 'r') as stream:
            json_dict = json.load(stream)
            return json_dict

    except ValueError as exc:
        # logger.error(str(exc))
        # logger.error(traceback.format_exc())
        print(str(exc))
        print(traceback.format_exc())        
        r_value = -1
    except BaseException:
        # logger.error(traceback.format_exc())
        print(traceback.format_exc())

    return r_value


def dates_definitions(start_date, end_date, N_days_X, N_days_y):
    '''
    This function defines all the dates in any needed format for the project
    :param (string) start_date: the date of the end of the time series (string format: YYYY-MM-DD)
    :param (string) end_date: the date of the end of the time series (string format: YYYY-MM-DD)
    :param (int) N_days_X: the number of days to be used for prediction
    :param (int) N_days_y: the number of days to be predicted
    :return (tuple): Event timepoints (datetime)
             - end_date_dt              : the date of the end of the time series (datetime format)
             - start_date_prediction    : the date of the start of the time series used for model training (string format: YYYY-MM-DD)
             - end_date_prediction      : the date of the end of the time series used for model training (string format: YYYY-MM-DD)
             - start_date_prediction_dt : the date of the start of the time series used for model training (datetime format)
             - end_date_prediction_dt   : the date of the end of the time series used for model training (datetime format)
             - start_date_prediction_dt : the date of the start of the time series used for model training (datetime format)
             - end_date_prediction_dt   : the date of the end of the time series used for model training (datetime format)
             - start_date_X_dt          : the date of the start of the X time series (datetime format)
             - end_date_X_dt            : the date of the end of the X time series (datetime format)
             - start_date_y_dt          : the date of the start of the y time series (datetime format)
             - end_daten_y_dt           : the date of the end of the y time series (datetime format)
             - end_date_plusOneDay      : the date of the end of the time series, plus one day (string format: YYYY-MM-DD)
             - end_date_minus_6month    : the date of the end of the time series, minus 6 months (datetime format)
    '''
    start_date_dt, end_date_dt = datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")

    start_date_prediction_dt = end_date_dt - relativedelta(days=N_days_X + N_days_y - 1)
    start_date_prediction = start_date_prediction_dt.strftime("%Y-%m-%d")
    end_date_prediction = end_date

    end_date_plusOneDay = datetime.strptime(end_date, "%Y-%m-%d")
    end_date_plusOneDay = end_date_plusOneDay + relativedelta(days=1)
    end_date_plusOneDay = end_date_plusOneDay.strftime("%Y-%m-%d")

    end_date_minus_6month = end_date_dt - relativedelta(months=6)

    return (
        start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay,
        end_date_minus_6month)


def time_series_generator(size=500,
                          cycle_period=30.5,
                          signal_type='sine',
                          salary=1,
                          trend=0.1,
                          noise=0.1,
                          offset=False,
                          spike=0):
    '''
    This function generates mock time series with noise
    :param (int) size: length of the time series
    :param (float) cycle_period: period of the signal (usually 30.5, the month period, in days)
    :param (string) signal_type: Type of signal, "sine", "sawtooth", "triangle", "square", "random_choice"
    :param (float) salary: Base scaling variable for the trend, default=1
    :param (float) trend: Scaling variable for the trend
    :param (float) noise: Trend noise, default=0.1
    :param (boolean) offset: Use of random phase offset, makes seasonality
    :param (int) spike: Number of random amplitude spikes
    :return (numpy array): Timeseries with account balance for each day
    '''

    signal_types = ['sine', 'sawtooth', 'triangle', 'square']
    if signal_type == 'random_choice':
        signal_type = random.choice(signal_types)
    elif signal_type not in signal_types:
        raise ValueError('{} is not a valid signal type'.format(signal_type))

    # in size = 635, and cycle_period = 30.5, we have ~ 21 periods (20.8)
    count_periods = size / cycle_period

    # 1. The trend making
    t = np.linspace(-0.5 * cycle_period * count_periods, 0.5 * cycle_period * count_periods, size)
    t_trend = np.linspace(0, 1, size)
    sign = random.choice([-1, 1])
    trend_ts = sign * salary * np.exp(trend*t_trend)

    # 2. The seasonality making
    if offset:
        phase = np.random.uniform(-1, 1) * np.pi
    else:
        phase = 0

    if signal_type == 'sine':     ts = 0.5 * salary * np.sin(2 * np.pi * (1. / cycle_period) * t + phase)
    if signal_type == 'sawtooth': ts = -0.5 * salary * signal.sawtooth(2 * np.pi * (1. / cycle_period) * t + phase)
    if signal_type == 'triangle': ts = 1 * salary * np.abs(signal.sawtooth(2 * np.pi * (1. / cycle_period) * t + phase)) - 1
    if signal_type == 'square':   ts = 0.5 * salary * signal.square(2 * np.pi * (1. / cycle_period) * t + phase)

    # 3. The noise making
    noise_ts = np.random.normal(0, noise * salary, size)

    ts = ts + trend_ts + noise_ts

    # 4. Adding spikes to the time series
    if spike > 0:
        last_spike_time = int(size)-92      # Don't create spikes in the last 3 months, where we want to predict
        first_spike_time = int(size)-92-365 # Let's have the spikes within 1 year up to the prediction time
        for _ in range(spike):
            sign = random.choice([-1, 1])
            t_spike = np.random.randint(first_spike_time, last_spike_time)  # time of the spike
            ts[t_spike:] = ts[t_spike:] + sign * np.random.normal(3 * salary, salary)
            print(t_spike)
            
    print(size, first_spike_time, last_spike_time)
            
    if signal_type == 'sine':     signal_type_int = 1
    if signal_type == 'triangle': signal_type_int = 2
    if signal_type == 'square':   signal_type_int = 3     
    if signal_type == 'sawtooth': signal_type_int = 4      

    return np.around(ts,decimals=2).tolist(), signal_type_int    


def pre_processing(ts_balance, end_date, spark, serving=False):
    '''
    This function performs several pre-processing tasks before TRAINING:
    - scales the time series (standardization: mean=0, std=1)
    - extrapolates linearly (first order extrapolation) the trend to the next 3 months (92 days)
    :param (spark dataframe) ts_balance:  the spark dataframe containing time series with the column 'balance_detrend_1MW'
    :param (string) end_date: the end date ('YYYY-MM-DD' string format)
    :param (spark instance) spark: the spark instance
    :param (boolean) serving: True if processing data for serving (skips the 'y' column)
    :return (spark dataframe): the same input dataframe with added columns:
               - mean                           : the mean of the time series. Used for scaling the time series
               - std                            : the standard deviation of the time series. Used for scaling the time series
               - balance_detrend_1MW_scaled     : the time series scaled
               - X                              : the time series part to be trained on
               - y                              : the time series part to be predicted (92 days, 3 months) NOT USED IN SERVING MODE!
               - trend_next_3months_1MW         : the linear extrapolation of the trend up to next 3 months
    '''

    # @F.pandas_udf("array<float>", PandasUDFType.SCALAR)
    # def trend2(x,window_size_days): #, window_size_days
    #     '''
    #     This function computes the trend of a time series x, using a time window of size window_size_days
    #     :param (list) x: Timeseries to operate on
    #     :param (int) window_size_days: Window size for seasonality decomposition
    #     :return (list): Trend values, same length as x
    #     '''

    #     #decomposed_ts = seasonal_decompose(np.array(x), model='additive', freq=30, extrapolate_trend=1)
    #     #trend = np.around(decomposed_ts.trend, decimals=3)
    #     #return trend.tolist()
    #     #return x.apply(lambda v: np.around(seasonal_decompose(v, model='additive', freq=30, extrapolate_trend=1).trend, decimals=3) )
    #     return Series([np.around(seasonal_decompose(np.array(c1), model='additive', freq=c2, extrapolate_trend=1).trend, decimals=3) for c1,c2 in zip(x,window_size_days)])

    # @F.pandas_udf("array<float>", PandasUDFType.SCALAR)
    # def detrend2(x,trend):
    #     '''
    #     This function subtracts a given trend from a timeseries
    #     :param (list) x: Timeseries to operate on
    #     :param (list) trend: The trend to be subtracted
    #     :return (list): Detrended timeseries
    #     '''
    #     #return np.around(np.array(x) - np.array(trend), decimals=3)
    #     return Series([np.around(np.array(c1) - np.array(c2), decimals=3) for c1,c2 in zip(x,trend)])

    # @F.pandas_udf("float", PandasUDFType.SCALAR)
    # def mean_for_scaling2(x):
    #     '''
    #     This function computes the mean of a time series x
    #     :param (list) x: The timeseries to compute the mean for
    #     :return (float): Mean value
    #     '''
    #     # TODO: Add a time window on the mean computation!
    #     return Series([np.around(np.mean(c1), decimals=3) for c1 in x])

    # @F.pandas_udf("float", PandasUDFType.SCALAR)
    # def std_for_scaling2(x):
    #     '''
    #     This function computes the standard deviation of a time series x
    #     :param (list) x: The timeseries to compute the std for
    #     :return (float): Standard deviation
    #     '''
    #     # TODO: Add a time window on the std computation!
    #     return Series([np.around(np.std(c1), decimals=3) for c1 in x])

    # @F.pandas_udf("array<float>", PandasUDFType.SCALAR)
    # def scaling2(x, mean, std):
    #     '''
    #     This function scales a time series x
    #     :param (list) x: Timeseries to be scaled
    #     :param (float) mean: The mean to be used for scaling
    #     :param (float) std: The standard deviation to be used for scaling
    #     :return (list): A scaled timeseries
    #     '''
    #     #scaled = np.around((np.array(x) - mean) / std, decimals=3)
    #     #return scaled.tolist()
    #     return Series([np.around((np.array(c1) - c2) / c3, decimals=3) for c1,c2,c3 in zip(x,mean,std)])

    # @F.pandas_udf("array<float>", PandasUDFType.SCALAR)
    # def get_X2(x, X_days, y_days):
    #     '''
    #     This function extracts the X part of train time series
    #     :param (list) x: Time series to be extracted from
    #     :param (int) X_days: number of days to be used for prediction
    #     :param (int) y_days: number of trailing days to skip
    #     :return (list): extracted timeseries
    #     '''
    #     #X_array = x[-X_days - y_days:-y_days]
    #     #X_array = np.around(X_array, decimals=3)
    #     #return X_array.tolist()
    #     return Series([np.around(c1[-c2-c3:-c3], decimals=3) for c1,c2,c3 in zip(x, X_days, y_days)])

    # @F.pandas_udf("array<float>", PandasUDFType.SCALAR)
    # def get_y2(x, y_days):
    #     '''
    #     This function extracts the y part of train time series
    #     :param (list) x: Time series to be extracted from
    #     :param (int) y_days: number of days to be used as y
    #     :return (list): extracted timeseries
    #     '''
    #     #y_array = x[-y_days:]
    #     #y_array = np.around(y_array, decimals=3)
    #     #return y_array.tolist()
    #     return Series([np.around(c1[-c2:], decimals=3) for c1,c2 in zip(x, y_days)])

    # trend computation
    ts_balance = ts_balance.withColumn('balance_trend_1MW',
                                       F.udf(lambda x, y: trend(x,y), "array<float>")('balance', F.lit(30)))
                                       #trend('balance', F.lit(30)))
    # detrend computation
    ts_balance = ts_balance.withColumn('balance_detrend_1MW',
                                       F.udf(lambda x, y: detrend(x,y), "array<float>")('balance', 'balance_trend_1MW'))
                                       #detrend('balance', 'balance_trend_1MW'))
    # scaling the detrended time series
    ts_balance = ts_balance.withColumn('mean',
                                       F.udf(lambda x: mean_for_scaling(x), "float")('balance_detrend_1MW'))
                                       #mean_for_scaling('balance_detrend_1MW'))
    ts_balance = ts_balance.withColumn('std',
                                       F.udf(lambda x: std_for_scaling(x), "float")('balance_detrend_1MW'))
                                       #std_for_scaling('balance_detrend_1MW'))
    ts_balance = ts_balance.withColumn("balance_detrend_1MW_scaled",
                                       F.udf(lambda x, y, z: scaling(x,y,z), "array<float>")('balance_detrend_1MW','mean','std'))
                                       #scaling('balance_detrend_1MW','mean','std'))
    ts_balance = ts_balance.withColumn('X',
                                       F.udf(lambda x, y, z: get_X(x,y,z), "array<float>")('balance_detrend_1MW_scaled', F.lit(365), F.lit(92)))
                                       #get_X('balance_detrend_1MW_scaled', F.lit(365), F.lit(92)))
    if not serving:
        ts_balance = ts_balance.withColumn('y',
                                           F.udf(lambda x, y: get_y(x,y), "array<float>")('balance_detrend_1MW_scaled', F.lit(92)))
                                           #get_y('balance_detrend_1MW_scaled', F.lit(92)))

    # Extrapolation of trend to next 3 months (92 days)
    # Need to have the end date + 1 day
    end_date_plusOneDay = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days=1)
    end_date_plusOneDay = end_date_plusOneDay.strftime("%Y-%m-%d")

    # extrapolation time
    end_date_plus92Day = datetime.strptime(end_date, "%Y-%m-%d") + relativedelta(days=92)
    end_date_plus92Day = end_date_plus92Day.strftime("%Y-%m-%d")
    extrapolated_spark_df = spark.sql("SELECT sequence(to_date('{0}'), to_date('{1}'), interval 1 day) as transactiondate_next3months".format(end_date_plusOneDay, end_date_plus92Day))

    # create all combinations of customers and dates, number_of_rows = n_customer x extrapolated_days(92)
    ts_balance = ts_balance.crossJoin(extrapolated_spark_df)

    ts_balance = ts_balance.withColumn('trend_next_3months_1MW',
                                       F.udf(lambda aa,bb,cc,dd,ee: extrapolate_trend(aa,bb,cc,dd,ee), "array<float>")('balance_trend_1MW', F.lit(183), F.lit(92), F.lit(True), F.lit(serving)))
                                       #extrapolate_trend('balance_trend_1MW', F.lit(183), F.lit(92), F.lit(True), F.lit(serving)))
    # trajectory metric
    if serving:
        ts_balance = ts_balance.withColumn('trajectory_6months',
                                           F.udf(lambda x,y,z: trajectory(x,y,z), "array<float>")('balance_trend_1MW', F.lit(6), F.lit(1)))
                                           #trajectory('balance_trend_1MW', F.lit(6), F.lit(1)))
    return ts_balance


#@F.udf("array<float>")
def trend(x,window_size_days):
    '''
    This function computes the trend of a time series x, using a time window of size window_size_days
    :param (list) x: Timeseries to operate on
    :param (int) window_size_days: Window size for seasonality decomposition
    :return (list): Trend values, same length as x
    '''
    decomposed_ts = seasonal_decompose(np.array(x), model='additive', freq=window_size_days, extrapolate_trend=1)
    trend = np.around(decomposed_ts.trend, decimals=3)
    return trend.tolist()

#@F.udf("array<float>")
def detrend(x, trend):
    '''
    This function subtracts a given trend from a timeseries
    :param (list) x: Timeseries to operate on
    :param (list) trend: The trend to be subtracted
    :return (list): Detrended timeseries
    '''
    detrend = np.array(x) #np.around(np.array(x) - np.array(trend), decimals=3)
    return detrend.tolist()


#@F.udf("array<float>")
def retrend(x, trend):
    '''
    This function adds a given trend to a timeseries
    :param (list) x: Timeseries to operate on
    :param (list) trend: The trend to be added
    :return (list): Retrended timeseries
    '''
    retrend = np.array(x) #np.around(np.array(x) + np.array(trend), decimals=3)
    return retrend.tolist()

#@F.udf("float")
def mean_for_scaling(x):
    '''
    This function computes the mean of a time series x
    :param (list) x: The timeseries to compute the mean for
    :return (float): Mean value
    '''
    # TODO: Add a time window on the mean computation!
    return float(np.around(np.mean(x), decimals=3))

#@F.udf("float")
def std_for_scaling(x):
    '''
    This function computes the standard deviation of a time series x
    :param (list) x: The timeseries to compute the std for
    :return (float): Standard deviation
    '''
    # TODO: Add a time window on the std computation!
    return float(np.around(np.std(x), decimals=3))

#@F.udf("array<float>")
def scaling(x, mean, std):
    '''
    This function scales a time series x
    :param (list) x: Timeseries to be scaled
    :param (float) mean: The mean to be used for scaling
    :param (float) std: The standard deviation to be used for scaling
    :return (list): A scaled timeseries
    '''
    scaled = np.around((np.array(x) - mean) / std, decimals=3)
    return scaled.tolist()

#@F.udf("array<float>")
def rescaling(x, mean, std):
    '''
    This function rescales a time series x
    :param (list) x: Timeseries to be rescaled
    :param (float) mean: The mean to be used for rescaling
    :param (float) std: The standard deviation to be used for rescaling
    :return (list): A rescaled timeseries
    '''

    rescaled = np.around((np.array(x) * std) + mean, decimals=3)
    return rescaled.tolist()


#@F.udf("array<float>")
def get_X(x, X_days, y_days):
    '''
    This function extracts the X part of train time series
    :param (list) x: Time series to be extracted from
    :param (int) X_days: number of days to be used for prediction
    :param (int) y_days: number of trailing days to skip
    :return (list): extracted timeseries
    '''

    X_array = x[-X_days - y_days:-y_days]
    X_array = np.around(X_array, decimals=3)
    return X_array.tolist()

#@F.udf("array<float>")
def get_y(x, y_days):
    '''
    This function extracts the y part of train time series
    :param (list) x: Time series to be extracted from
    :param (int) y_days: number of days to be used as y
    :return (list): extracted timeseries
    '''

    y_array = x[-y_days:]
    y_array = np.around(y_array, decimals=3)
    return y_array.tolist()

#@F.udf("array<float>")
def extrapolate_trend(x, window_size_days, extrapolation_window_size_days, median=True, serving_mode=False):
    '''
    This function extrapolates the trend for time series x
    :param (list) x: Timeseries to be operated on
    :param (int) window_size_days: number of days to be used for calculation of the trend
    :param (int) extrapolation_window_size_days: number of days for extrapolation
    :param (boolean) median: should median be used to calculate the aggregate slope, default is True
    :param (boolean) serving_mode: if we are in the serving mode, default is False
    :return (list):  extrapolated trend
    '''

    slope = np.gradient(x)

    if not serving_mode:  # if mode is "train"
        if median:
            aggregated_slope = np.nanmedian(slope[-window_size_days - extrapolation_window_size_days:])
        else:
            aggregated_slope = np.nanmean(slope[-window_size_days - extrapolation_window_size_days:])
    else:  # if mode is "serve"
        if median:
            aggregated_slope = np.nanmedian(slope[-window_size_days:])
        else:
            aggregated_slope = np.nanmean(slope[-window_size_days:])

    # we build the extrapolation
    trend_extrapolate = np.zeros(extrapolation_window_size_days)
    if not serving_mode:  # if mode is "train"
        trend_extrapolate[0] = aggregated_slope + x[-1 - extrapolation_window_size_days]
    else:
        trend_extrapolate[0] = aggregated_slope + x[-1]

    for day in range(1, extrapolation_window_size_days):
        trend_extrapolate[day] = aggregated_slope + trend_extrapolate[day - 1]

    trend_extrapolate = np.around(trend_extrapolate, decimals=3)
    return trend_extrapolate.tolist()

#@F.udf("float")
def trajectory(x, window_size_months, median=True):
    '''
    This function computes the customer "trajectory", which is computed as the median/mean value
    of the gradients of the time series trend, over the last window_size_months.
    If median is True gives median of slope, otherwise gives mean of slope
    :param (list) x: Timeseries to be operated on
    :param (int) window_size_months: number of months to be used for calculation
    :param (boolean) median: should median be used to calculate the aggregate slope, default is True
    :return (float): the aggregated slope fo the given time window
    '''

    slope = np.gradient(x)
    n_days = int(30.5 * window_size_months)

    # we take the median of the daily slope
    if median:
        aggregated_slope = np.nanmedian(slope[-n_days:])
    else:
        aggregated_slope = np.nanmean(slope[-n_days:])
    result = aggregated_slope * 30.5  # we scale it to a month slope #TODO: why multiplying by 30.5?
    return float(np.around(result, decimals=3))


# def post_processing(ts_balance):
#     '''
#     This function performs several post-processing tasks:
#     - scales the predicted time series to the original scale of the time series
#     - re-trends the predicted time series. This means that the predicted time series (which has zero trend
#       at the output of the forecast method) is summed with the trend (extrapolated) of the time series
#     :param (spark dataframe) ts_balance: the spark dataframe containing time series with the column
#     :return (spark dataframe): the same input dataframe with added columns:
#                - y_pred_rescaled           : the predicted time series is scaled to the original scale
#                - y_pred_rescaled_retrended : the predicted time series is re-trended
#     '''

#     ts_balance = ts_balance.withColumn("y_pred_rescaled",
#                                        F.udf(lambda x, y, z: rescaling(x,y,z), "array<float>")(
#                                        #rescaling(
#                                             'y_pred',
#                                             'mean',
#                                             'std'))

#     ts_balance = ts_balance.withColumn('y_pred_rescaled_retrended',
#                                        F.udf(lambda x, y: retrend(x,y), "array<float>")(
#                                        #retrend(
#                                             'y_pred_rescaled',
#                                             'trend_next_3months_1MW'))
#     return ts_balance


# def define_1dcnn_model(N_days_X, N_days_y, model_conf):
#     '''
#     :param N_days_X: Number of days for training
#     :param N_days_y: Number of days to be predicted
#     :param (dict) model_conf: dictionary of model hyperparameters
#     :return (keras model): A compiled 1DCNN keras model
#     '''

#     hyperparameters = model_conf['hyperParameters']

#     opt = tf.compat.v1.train.AdamOptimizer()

#     # tf.keras, Functional API
#     inputs = tf.keras.layers.Input(shape=(N_days_X, 1,), name='input')
#     x = tf.keras.layers.Conv1D(filters=int(hyperparameters['filters']), kernel_size=int(hyperparameters['kernel_size']),
#                                activation=hyperparameters['activation'])(inputs)
#     x = tf.keras.layers.MaxPooling1D(pool_size=int(hyperparameters['pool_size']))(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(int(hyperparameters['dense_units']), activation=hyperparameters['activation'])(x)
#     outputs = tf.keras.layers.Dense(N_days_y, name='output')(x)
#     model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

#     # Model compilation
#     model.compile(optimizer=opt, loss=hyperparameters['loss'])

#     return model


# def metric_extraction(df, N_days_y):
#     '''
#     This function extracts the R2 metric for the prediction on the test set
#     :param (pandas df) df: the dataframe containing the time series to be visualized
#     :param (int) N_days_y: the size of the predicted array
#     :return:
#              - R2_all_3month (float)   : R2 metric computed over 3 months prediction time
#              - R2_array_3month (array) : R2 metric computed over 3 months prediction time,
#                                          computed for each time series
#              - R2_all_1month (float)   : R2 metric computed over 1 months prediction time
#              - R2_array_1month (array) : R2 metric computed over 1 months prediction time,
#                                          computed for each time series
#     '''
    
#     # Extraction of the arrays from the time series format dataframe
#     for column in df.columns:
#         df[column] = df[column].apply(lambda x: np.array(x))

#     balance = np.array(df['balance'].tolist())
#     balance_y_pd = pd.DataFrame(balance[:, -N_days_y:]).T

#     y_pred_f_pd = pd.DataFrame(np.array(df['y_pred_rescaled_retrended'].tolist())).T

#     y_test_ready = balance_y_pd.iloc[:, 1:].values
#     y_pred = y_pred_f_pd.iloc[:, 1:].values

#     # R2 at 3 months
#     R2_all_3month = r2_score(y_test_ready, y_pred, multioutput='uniform_average')
#     R2_array_3month = r2_score(y_test_ready, y_pred,
#                                multioutput='raw_values')  # Computing R2 for individual time series

#     # R2 at 1 month
#     R2_all_1month = r2_score(y_test_ready[:, 0:31], y_pred[:, 0:31], multioutput='uniform_average')
#     R2_array_1month = r2_score(y_test_ready[:, 0:31], y_pred[:, 0:31],
#                                multioutput='raw_values')  # Computing R2 for individual time series

#     return R2_all_3month, R2_array_3month, R2_all_1month, R2_array_1month


# def visualization_prediction(df, start_date, end_date, N_days_X, N_days_y, R2_array_1month, R2_array_3month, serving=False):
#     '''
#     This function produces a visualization of the prediction for a few time series
#     If mode serving enabled, if also visualizes the distribution of R2 (TODO)
#     :param (pandas df) df: the dataframe containing the time series to be visualized
#     :param (string) start_date: the date of the end of the time series (string format: YYYY-MM-DD)
#     :param (string) end_date: the date of the end of the time series (string format: YYYY-MM-DD)
#     :param (int) N_days_X: size of array used for prediction
#     :param (int) N_days_y: size of array predicted
#     :param (boolean) serving: True if processing data for serving (skips the 'y' column)
#     '''

#     for column in df.columns:
#         df[column] = df[column].apply(lambda x: np.array(x))

#     balance = np.array(df['balance'].tolist())
#     balance_pd = pd.DataFrame(balance).T
#     balance_y_pd = pd.DataFrame(balance[:, -N_days_y:]).T
#     balance_X_pd = pd.DataFrame(balance[:, -N_days_y - N_days_X:-N_days_y]).T

#     X_test_pd = pd.DataFrame(np.array(df['X'].tolist())).T

#     if not serving:
#         y_test_pd = pd.DataFrame(np.array(df['y'].tolist())).T

#     y_pred_pd = pd.DataFrame(np.array(df['y_pred'].tolist())).T
#     y_pred_f_pd = pd.DataFrame(np.array(df['y_pred_rescaled_retrended'].tolist())).T

#     y_test_ready = balance_y_pd.iloc[:, 1:].values
#     y_pred = y_pred_f_pd.iloc[:, 1:].values

#     time_all = pd.date_range(start_date, end_date, freq='D')
#     if not serving:
#         time_y = time_all[-N_days_y:]
#         time_X = time_all[-N_days_y - N_days_X:-N_days_y]
#     else:
#         time_y = pd.date_range(start=end_date, periods=N_days_y + 1, freq='D')[1:]
#         time_X = time_all[-N_days_X:]

#     #balance_pd = balance_pd.set_index(time_all).reset_index()
#     balance_y_pd = balance_y_pd.set_index(time_y).reset_index()
#     balance_X_pd = balance_X_pd.set_index(time_X).reset_index()
#     X_test_pd = X_test_pd.set_index(time_X).reset_index()
#     if not serving:
#         y_test_pd = y_test_pd.set_index(time_y).reset_index()
#     y_pred_pd = y_pred_pd.set_index(time_y).reset_index()
#     y_pred_f_pd = y_pred_f_pd.set_index(time_y).reset_index()

#     #balance_pd.rename(columns={'index': 'date'}, inplace=True)
#     balance_y_pd.rename(columns={'index': 'date'}, inplace=True)
#     balance_X_pd.rename(columns={'index': 'date'}, inplace=True)
#     X_test_pd.rename(columns={'index': 'date'}, inplace=True)
#     if not serving:
#         y_test_pd.rename(columns={'index': 'date'}, inplace=True)
#     y_pred_pd.rename(columns={'index': 'date'}, inplace=True)
#     y_pred_f_pd.rename(columns={'index': 'date'}, inplace=True)

#     #balance_pd['date'] = pd.to_datetime(balance_pd['date'])
#     balance_y_pd['date'] = pd.to_datetime(balance_y_pd['date'])
#     balance_X_pd['date'] = pd.to_datetime(balance_X_pd['date'])
#     X_test_pd['date'] = pd.to_datetime(X_test_pd['date'])
#     if not serving:
#         y_test_pd['date'] = pd.to_datetime(y_test_pd['date'])
#     y_pred_pd['date'] = pd.to_datetime(y_pred_pd['date'])
#     y_pred_f_pd['date'] = pd.to_datetime(y_pred_f_pd['date'])

#     # Figure of a few time series
#     fig1 = plt.figure(1, (25, 15))

#     columns = X_test_pd.columns[1:10]  # TODO select 10 random time series!

#     ax = plt.subplot(2, 1, 1)
#     for column in columns:
#         plt.plot(X_test_pd.iloc[:, 0], X_test_pd[column], 'o--')
#         if not serving:
#             plt.plot(y_test_pd.iloc[:, 0], y_test_pd[column], 'o--')
#         plt.plot(y_pred_pd.iloc[:, 0], y_pred_pd[column], 'ro--')

#     ax.axhline(y=0., color='gray', linestyle='--')
#     ax.set_xlabel('Date', fontsize=20)
#     ax.set_ylabel('Balance', fontsize=20)

#     ax = plt.subplot(2, 1, 2)
#     for column in columns:
#         #plt.plot(balance_pd.iloc[:, 0], balance_pd[column], 'bo--')
#         plt.plot(balance_X_pd.iloc[:, 0], balance_X_pd[column], 'bo--')
#         plt.plot(balance_y_pd.iloc[:, 0], balance_y_pd[column], 'bo--')
#         plt.plot(y_pred_f_pd.iloc[:, 0], y_pred_f_pd[column], 'ro--')

#     ax.axhline(y=0., color='gray', linestyle='--')
#     ax.set_xlabel('Date', fontsize=20)
#     ax.set_ylabel('Balance', fontsize=20)

#     #plt.show()
#     #fig1.savefig('performance.png')
#     #display(fig2)

#     # Figure of R2 (only in case of evaluation, not for serving)
#     if not serving:
#         fig2 = plt.figure(2, (25, 15))

#         # Creating a squeezed version of R2, in the range [0,1]
#         def sigmoid_squeezed(x):
#             return (1 + np.exp(-1)) / (1 + np.exp(-x))

#         R2_array_3month_squeezed = sigmoid_squeezed(R2_array_3month)
#         R2_array_1month_squeezed = sigmoid_squeezed(R2_array_1month)

#         ax1 = plt.subplot(2, 2, 1)
#         plt.hist(R2_array_3month, bins=20, range=[-2, 1],
#                  histtype='step', color='red',
#                  label='R2 3-month', density=True)
#         plt.hist(R2_array_1month, bins=20, range=[-2, 1],
#                  histtype='step', color='blue',
#                  label='R2 1-month', density=True)
#         ax1.set_xlabel('R square', fontsize=15)
#         ax1.set_ylabel('N', fontsize=15)

#         ax1 = plt.subplot(2, 2, 2)
#         plt.hist(R2_array_3month_squeezed, bins=20, range=[0, 1],
#                  histtype='step', color='red',
#                  label='R2 3-month', density=True)
#         plt.hist(R2_array_1month_squeezed, bins=20, range=[0, 1],
#                  histtype='step', color='blue',
#                  label='R2 1-month', density=True)
#         ax1.set_xlabel('R square squeezed', fontsize=15)
#         ax1.set_ylabel('N', fontsize=15)

#         plt.legend(loc='upper right', fontsize=15)
#         #plt.show()
#         #fig2.savefig('performance_R2.png')
#         #display(fig2)
#     if serving: fig2 = 0        
        
#     return fig1, fig2
        
        
# def visualization_time_series_pred_only(df, start_date, end_date, N_days_X, N_days_y, R2_array_1month, R2_array_3month, serving=False):
#     '''
#     This function produces a visualization of the prediction for a few time series
#     If mode serving enabled, if also visualizes the distribution of R2 (TODO)
#     :param (pandas df) df: the dataframe containing the time series to be visualized
#     :param (string) start_date: the date of the end of the time series (string format: YYYY-MM-DD)
#     :param (string) end_date: the date of the end of the time series (string format: YYYY-MM-DD)
#     :param (int) N_days_X: size of array used for prediction
#     :param (int) N_days_y: size of array predicted
#     :param (boolean) serving: True if processing data for serving (skips the 'y' column)
#     '''

#     for column in df.columns:
#         df[column] = df[column].apply(lambda x: np.array(x))

#     balance = np.array(df['balance'].tolist())
#     balance_pd = pd.DataFrame(balance).T
#     balance_y_pd = pd.DataFrame(balance[:, -N_days_y:]).T
#     balance_X_pd = pd.DataFrame(balance[:, -N_days_y - N_days_X:-N_days_y]).T    

#     X_test_pd = pd.DataFrame(np.array(df['X'].tolist())).T

#     if not serving:
#         y_test_pd = pd.DataFrame(np.array(df['y'].tolist())).T

#     y_pred_pd = pd.DataFrame(np.array(df['y_pred'].tolist())).T
#     y_pred_f_pd = pd.DataFrame(np.array(df['y_pred_rescaled_retrended'].tolist())).T

#     y_test_ready = balance_y_pd.iloc[:, 1:].values
#     y_pred = y_pred_f_pd.iloc[:, 1:].values

#     time_all = pd.date_range(start_date, end_date, freq='D')
#     if not serving:
#         time_y = time_all[-N_days_y:]
#         time_X = time_all[-N_days_y - N_days_X:-N_days_y]
#     else:
#         time_y = pd.date_range(start=end_date, periods=N_days_y + 1, freq='D')[1:]
#         time_X = time_all[-N_days_X:]

#     #balance_pd = balance_pd.set_index(time_all).reset_index()
#     balance_y_pd = balance_y_pd.set_index(time_y).reset_index()
#     balance_X_pd = balance_X_pd.set_index(time_X).reset_index()
#     X_test_pd = X_test_pd.set_index(time_X).reset_index()
#     if not serving:
#         y_test_pd = y_test_pd.set_index(time_y).reset_index()
#     y_pred_pd = y_pred_pd.set_index(time_y).reset_index()
#     y_pred_f_pd = y_pred_f_pd.set_index(time_y).reset_index()

#     #balance_pd.rename(columns={'index': 'date'}, inplace=True)
#     balance_y_pd.rename(columns={'index': 'date'}, inplace=True)
#     balance_X_pd.rename(columns={'index': 'date'}, inplace=True)
#     X_test_pd.rename(columns={'index': 'date'}, inplace=True)
#     if not serving:
#         y_test_pd.rename(columns={'index': 'date'}, inplace=True)
#     y_pred_pd.rename(columns={'index': 'date'}, inplace=True)
#     y_pred_f_pd.rename(columns={'index': 'date'}, inplace=True)

#     #balance_pd['date'] = pd.to_datetime(balance_pd['date'])
#     balance_y_pd['date'] = pd.to_datetime(balance_y_pd['date'])
#     balance_X_pd['date'] = pd.to_datetime(balance_X_pd['date'])
#     X_test_pd['date'] = pd.to_datetime(X_test_pd['date'])
#     if not serving:
#         y_test_pd['date'] = pd.to_datetime(y_test_pd['date'])
#     y_pred_pd['date'] = pd.to_datetime(y_pred_pd['date'])
#     y_pred_f_pd['date'] = pd.to_datetime(y_pred_f_pd['date'])

#     columns = X_test_pd.columns[1:10]  # TODO select 10 random time series!

#     fig1 = plt.figure(1, (15, 7))
#     ax = plt.subplot(111)
#     for column in columns:
#         #plt.plot(balance_pd.iloc[:, 0], balance_pd[column], 'b-')
#         plt.plot(balance_X_pd.iloc[:, 0], balance_X_pd[column], 'b-')
#         plt.plot(balance_y_pd.iloc[:, 0], balance_y_pd[column], 'b-')
#         plt.plot(y_pred_f_pd.iloc[:, 0], y_pred_f_pd[column], 'r--')
        
#     ax.axhline(y=0., color='gray', linestyle='--')
#     ax.set_xlabel('Date', fontsize=20)
#     ax.set_ylabel('Balance', fontsize=20)
    
#     ax.axhline(y=0., color='gray', linestyle='--',alpha=0.5)
#     ax.set_xlabel('Date', fontsize=20)
#     ax.set_ylabel('Balance', fontsize=20)
    
#     end_date = datetime.strptime(end_date,'%Y-%m-%d')
#     now_moment = end_date - relativedelta(months=3)
#     date_1months_ago = now_moment - relativedelta(months=1)
#     date_2months_ago = now_moment - relativedelta(months=2)
#     date_3months_ago = now_moment - relativedelta(months=3)
#     date_4months_ago = now_moment - relativedelta(months=4)
#     date_5months_ago = now_moment - relativedelta(months=5)
#     date_6months_ago = now_moment - relativedelta(months=6)
#     date_7months_ago = now_moment - relativedelta(months=7)
#     date_8months_ago = now_moment - relativedelta(months=8)
#     date_9months_ago = now_moment - relativedelta(months=9)
#     date_10months_ago = now_moment - relativedelta(months=10)
#     date_11months_ago = now_moment - relativedelta(months=11)
#     date_12months_ago = now_moment - relativedelta(months=12)
#     date_13months_ago = now_moment - relativedelta(months=13)
#     ax.axvline(date_12months_ago, color='grey', linestyle='--')
#     ax.axvline(now_moment, color='grey', linestyle='--')  
#     #ax.axvspan(date_12months_ago,end_date, color='orange', alpha=0.2) 
    
#     date_in_1months = now_moment + relativedelta(months=1) 
#     date_in_2months = now_moment + relativedelta(months=2) 
#     date_in_3months = now_moment + relativedelta(months=3) 
#     date_in_4months = now_moment + relativedelta(months=4) 
#     #ax.axvline(date_in_3months, color='grey', linestyle='--')
     
#     ax.axvline(date_12months_ago, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_11months_ago, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_10months_ago, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_9months_ago, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_8months_ago, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_7months_ago, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_6months_ago, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_5months_ago, color='grey', linestyle='--',alpha=0.2)    
#     ax.axvline(date_4months_ago, color='grey', linestyle='--',alpha=0.2) 
#     ax.axvline(date_3months_ago, color='grey', linestyle='--',alpha=0.2) 
#     ax.axvline(date_2months_ago, color='grey', linestyle='--',alpha=0.2) 
#     ax.axvline(date_1months_ago, color='grey', linestyle='--',alpha=0.2)     
#     ax.axvline(end_date, color='grey', linestyle='--')  
#     #ax.axvspan(date_6months_ago,end_date, color='orange', alpha=0.2)

#     ax.axvline(date_in_1months, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_in_2months, color='grey', linestyle='--',alpha=0.2)
#     ax.axvline(date_in_3months, color='grey', linestyle='--')
#     ax.axvspan(now_moment,date_in_3months, color='orange', alpha=0.5)    
    
#     ax.text(0.3, 0.92, 'past', transform=ax.transAxes,fontsize=20,color='black')
#     ax.text(0.85, 0.92, 'prediction', transform=ax.transAxes,fontsize=20,color='black')
    
#     ax.set_xlim([date_12months_ago,date_in_3months])
    
#     #ax.set_xticklabels([])
#     #ax.set_yticklabels([])    
#     #ax.set_xticks([])
#     #ax.set_yticks([])        

#     #plt.show()
#     #fig1.savefig('performance.png')
#     #display(fig1)

#     # Figure of R2 (only in case of evaluation, not for serving)
#     if not serving:
#         fig2 = plt.figure(2, (25, 15))

#         # Creating a squeezed version of R2, in the range [0,1]
#         def sigmoid_squeezed(x):
#             return (1 + np.exp(-1)) / (1 + np.exp(-x))

#         R2_array_3month_squeezed = sigmoid_squeezed(R2_array_3month)
#         R2_array_1month_squeezed = sigmoid_squeezed(R2_array_1month)

#         ax1 = plt.subplot(2, 2, 1)
#         plt.hist(R2_array_3month, bins=20, range=[-2, 1],
#                  histtype='step', color='red',
#                  label='R2 3-month', density=True)
#         plt.hist(R2_array_1month, bins=20, range=[-2, 1],
#                  histtype='step', color='blue',
#                  label='R2 1-month', density=True)
#         ax1.set_xlabel('R square', fontsize=15)
#         ax1.set_ylabel('N', fontsize=15)

#         ax1 = plt.subplot(2, 2, 2)
#         plt.hist(R2_array_3month_squeezed, bins=20, range=[0, 1],
#                  histtype='step', color='red',
#                  label='R2 3-month', density=True)
#         plt.hist(R2_array_1month_squeezed, bins=20, range=[0, 1],
#                  histtype='step', color='blue',
#                  label='R2 1-month', density=True)
#         ax1.set_xlabel('R square squeezed', fontsize=15)
#         ax1.set_ylabel('N', fontsize=15)

#         plt.legend(loc='upper right', fontsize=15)
#         #plt.show()
#         #fig2.savefig('performance_R2.png')
#         #display(fig2) 
#     if serving: fig2 = 0
        
#     return fig1, fig2   
  

def time_series_cleaning(x,threshold_count):
    '''
    This function removes the time series which are 
    - (option 1) made only of zeroes, or
    - (option 2) made of too little changes, below a threshold defined (threshold_count)
    It is indeed sometimes unuseful (and harmful for convergence) to train autoencoder
    
    on time series containing little information
    :param (list) x: Timeseries to operate on
    :param (int) threshold_count:the threshold of number of changes below which a time series is removed
    :return (int) keep_ts: label that says if the time series is to be kept (1) or not (0)
    '''

    x = np.array(x)

    if np.all(x == x[0]): # test if all values are the same (i.e. no transactions in the array)
        return 0
    elif np.diff(x).astype(bool).sum(axis=0) < threshold_count: # This counts the non-zeros CHANGES (diff) in the ts
        return 0
    else: return 1
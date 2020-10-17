# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

# ===============
# Packages import
# ===============

from __future__ import division
from datetime import datetime
import os
import random
import pandas as pd
import numpy as np
import logging
import yaml
import json
from dateutil.relativedelta import relativedelta
from scipy import signal
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql import Row
import pyspark.sql.types as pst
from pyspark.sql.functions import udf
    
# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "false")

# COMMAND ----------

# blob_name = "blob1"
# account_name = "aacdlml0461491171"
# cwd = "wasbs://"+blob_name+"@"+account_name+".blob.core.windows.net/"
# storageKey1 = dbutils.secrets.get(scope = "key-vault-secrets-cloudai", key = "storageaccountcloudaiKey1")
# spark.conf.set("fs.azure.account.key."+account_name+".blob.core.windows.net", storageKey1)

data_json = '''{
    "dev": {    
        "table_to_train_on": "cashflow_ts_mock_100K_ts_3spikes",
        "table_to_score": "cashflow_ts_mock_2p5M_ts_heldout",
        "table_scored": "cashflow_ts_mock_2p5M_ts_served_final",
        "cashflow_s1_out_scoring": "cashflow_ts_mock_2p5M_ts_s1",
        "cashflow_s2_out_scoring": "cashflow_ts_mock_2p5M_ts_s2",
        "table_test_for_performance": "cashflow_test_100K_ts_3spikes",
        "table_test_for_performance_scored": "cashflow_test_scored"     
    },  
    "prod": {    
        "table_to_train_on": "cashflow_ts_mock_100K_ts_3spikes",
        "table_to_score": "cashflow_ts_mock_2p5M_ts_heldout",
        "table_scored": "cashflow_ts_mock_2p5M_ts_served_final",
        "cashflow_s1_out_scoring": "cashflow_ts_mock_2p5M_ts_s1",
        "cashflow_s2_out_scoring": "cashflow_ts_mock_2p5M_ts_s2",
        "table_test_for_performance": "cashflow_test_100K_ts_3spikes",
        "table_test_for_performance_scored": "cashflow_test_scored"     
    }, 
    "synthetic_data": {    
        "table_to_train_on": "cashflow_ts_mock_100K_ts_3spikes",
        "table_to_score": "cashflow_ts_mock_2p5M_ts_heldout",
        "table_scored": "cashflow_ts_mock_2p5M_ts_served_final",
        "cashflow_s1_out_scoring": "cashflow_ts_mock_2p5M_ts_s1",
        "cashflow_s2_out_scoring": "cashflow_ts_mock_2p5M_ts_s2",
        "table_test_for_performance": "cashflow_test_100K_ts_3spikes",
        "table_test_for_performance_scored": "cashflow_test_scored"     
    },          
    "start_date": "2018-12-01", 
    "end_date": "2020-03-31",
    "number_of_historical_days": "365",
    "number_of_predicted_days": "92"
}'''

config_json = '''{
    "model_name": "tf_models_100K_ts_3spikes_dummy",
    "hyperParameters": {
        "filters": 64,
        "kernel_size": 2,
        "activation": "relu",
        "pool_size": "2",
        "dense_units": "50",
        "loss": "mae",
        "batch_size": 200,        
        "epochs": 15
    }
}'''

data_conf = json.loads(data_json)
model_conf = json.loads(config_json)
print(data_conf)
print(model_conf)

# COMMAND ----------

# ===========================
# Reading configuration files
# ===========================

# Reading configuration files
#data_conf = Get_Data_From_JSON(cwd + "data.json")
#model_conf = Get_Data_From_JSON(cwd + "config.json")

# data_conf = spark.read.option("multiline", "true").json(cwd+"data.json").toPandas()
# model_conf = spark.read.option("multiline", "true").json(cwd+"config.json").toPandas()

# data_conf =  json.loads(data_conf.to_json(orient='records'))[0]
# model_conf = json.loads(model_conf.to_json(orient='records'))[0]

start_date, end_date = data_conf['start_date'], data_conf['end_date']
N_days_X, N_days_y = int(data_conf['number_of_historical_days']), int(data_conf['number_of_predicted_days'])  # 365, 92

end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
start_date_for_prediction_dt = end_date_dt - relativedelta(days=N_days_X + N_days_y)
start_date_for_prediction = start_date_for_prediction_dt.strftime("%Y-%m-%d")

start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay, end_date_minus_6month = dates_definitions(
    start_date, end_date, N_days_X, N_days_y)

time_range = pd.date_range(start_date, end_date, freq='D')

# Type of dataset desired
# Case we want a dataset to train a model: use 1e5 and serving_mode=False
# Case we want an unseen dataset to serve the model on: use 2.5e6 and serving_mode=True
N_customers = 1e4 #2.5e6
serving_mode = True  # True if creating data for serving

# COMMAND ----------

# =========
# Functions
# =========

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

# COMMAND ----------

# ============================
# Generation of synthetic data (for both formats)
# ============================

dff = spark.range(N_customers).toDF("primaryaccountholder") #'primaryaccountholder','transactiondate','balance'

#@udf("array<float>") 
def ts_generation():
    bb,nn = time_series_generator(
              size=len(time_range),
              cycle_period=30.5,
              signal_type='random_choice',
              salary=np.maximum(np.random.normal(15000, 5000), 100),
              trend=np.random.uniform(1,2),#np.random.normal(0, 1.1),
              noise=np.abs(np.random.normal(0, 0.01)) + 0.1,
              offset=True,
              spike=3)      
    return Row('signal_type', 'balance')(nn, bb)
    
schema = pst.StructType([
    pst.StructField("signal_type", pst.IntegerType(), False),
    pst.StructField("balance", pst.ArrayType(pst.FloatType()), False)])    
    
ts_generation_udf = F.udf(ts_generation, schema)  
  
dff = dff.withColumn("generation", ts_generation_udf())

dff = dff.select('primaryaccountholder', "generation.*")

dff2 = spark.sql("SELECT sequence(to_date('{0}'), to_date('{1}'), interval 1 day) as transactiondate".format(start_date, end_date))

timeseries_spark = dff2.crossJoin(dff)
timeseries_spark = timeseries_spark.select('primaryaccountholder','transactiondate','balance','signal_type')

timeseries_spark.show(5)
timeseries_spark.count()

# COMMAND ----------

# ========================
# Saving the dataset
# ========================

# if not serving_mode:
#     table_out = data_conf['synthetic_data']['table_to_train_on'] 
# else:    
#     table_out = data_conf['synthetic_data']['table_to_score']
    
if not serving_mode:
    table_out = data_conf['dev']['table_to_train_on'] 
else:    
    table_out = data_conf['dev']['table_to_score']    

#timeseries_spark.write.format("parquet").mode("overwrite").save(cwd+"{0}.parquet".format(table_out)) 

#timeseries_spark.write.mode("overwrite").partitionBy("primaryaccountholder").saveAsTable(table_out) #too long to run...
#timeseries_spark.write.mode("overwrite").saveAsTable(table_out)

# write to delta: https://docs.databricks.com/delta/delta-batch.html#create-a-table
timeseries_spark.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format(table_out)) # can add .partitionby() but slow...

# COMMAND ----------

# read delta table or a table: https://docs.databricks.com/delta/delta-batch.html#read-a-table

#spark.table("events")    # query table in the metastore

bidule = spark.read.format("delta").load("/mnt/delta/{0}".format(table_out))  # query table by path
bidule.show(3)                                         

# COMMAND ----------

bidule.count()

# COMMAND ----------

# more on tables here https://docs.databricks.com/data/tables.html

# COMMAND ----------

# MAGIC %fs rm -r /user/hive/warehouse/cashflow_ts_mock_2p5m_ts_heldout

# COMMAND ----------

# MAGIC %fs ls /mnt/delta

# COMMAND ----------

# ==================================
# Visualization of a few time series
# ==================================

# Extracting the test set to Pandas
ts_balance_pd = timeseries_spark.limit(100).toPandas()

# COMMAND ----------

def visualization_time_series(df, start_date, end_date, N_days_X, N_days_y,limit=50000):
    '''
    This function produces a visualization for a few time series
    :param (pandas df) df: the dataframe containing the time series to be visualized
    :param (string) start_date: the date of the end of the time series (string format: YYYY-MM-DD)
    :param (string) end_date: the date of the end of the time series (string format: YYYY-MM-DD)
    :param (int) N_days_X: size of array used for prediction
    :param (int) N_days_y: size of array predicted
    '''
    
    list_signals = df['signal_type'].tolist()

    balance = np.array(df['balance'].tolist())
    balance_pd = pd.DataFrame(balance[:, -N_days_y - N_days_X:]).T
    balance_y_pd = pd.DataFrame(balance[:, -N_days_y:]).T
    balance_X_pd = pd.DataFrame(balance[:, -N_days_y - N_days_X:-N_days_y]).T
    
    time_all = pd.date_range(start_date, end_date, freq='D')
    time_y = time_all[-N_days_y:]
    time_X = time_all[-N_days_y - N_days_X:-N_days_y]

    #balance_pd = balance_pd.set_index(time_all).reset_index()
    balance_y_pd = balance_y_pd.set_index(time_y).reset_index()
    balance_X_pd = balance_X_pd.set_index(time_X).reset_index()
    
    #balance_pd.rename(columns={'index': 'date'}, inplace=True)
    balance_y_pd.rename(columns={'index': 'date'}, inplace=True)   
    balance_X_pd.rename(columns={'index': 'date'}, inplace=True) 
    
    #balance_pd['date'] = pd.to_datetime(balance_pd['date'])
    balance_y_pd['date'] = pd.to_datetime(balance_y_pd['date'])    
    balance_X_pd['date'] = pd.to_datetime(balance_X_pd['date']) 

    # Figure of a few time series
    fig = plt.figure(1, (15, 7))

    columns = balance_X_pd.columns[1:] 

    ax = plt.subplot(1,1,1)
    for i, column in enumerate(columns):
        if list_signals[i]==1: color = 'blue'
        if list_signals[i]==2: color = 'green'
        if list_signals[i]==3: color = 'orange'
        if list_signals[i]==4: color = 'red'
        plt.plot(balance_X_pd.iloc[:, 0], balance_X_pd[column], '-', lw=1,color=color)
        plt.plot(balance_y_pd.iloc[:, 0], balance_y_pd[column], '--', lw=1,color=color)


    ax.axhline(y=0., color='gray', linestyle='--',alpha=0.5)
    ax.set_xlabel('Date', fontsize=20)
    ax.set_ylabel('Balance', fontsize=20)
    
    end_date = datetime.strptime(end_date,'%Y-%m-%d')
    now_moment = end_date - relativedelta(months=3)
    date_1months_ago = now_moment - relativedelta(months=1)
    date_2months_ago = now_moment - relativedelta(months=2)
    date_3months_ago = now_moment - relativedelta(months=3)
    date_4months_ago = now_moment - relativedelta(months=4)
    date_5months_ago = now_moment - relativedelta(months=5)
    date_6months_ago = now_moment - relativedelta(months=6)
    date_7months_ago = now_moment - relativedelta(months=7)
    date_8months_ago = now_moment - relativedelta(months=8)
    date_9months_ago = now_moment - relativedelta(months=9)
    date_10months_ago = now_moment - relativedelta(months=10)
    date_11months_ago = now_moment - relativedelta(months=11)
    date_12months_ago = now_moment - relativedelta(months=12)
    date_13months_ago = now_moment - relativedelta(months=13)
    ax.axvline(date_12months_ago, color='grey', linestyle='--')
    ax.axvline(now_moment, color='grey', linestyle='--')  
    #ax.axvspan(date_12months_ago,end_date, color='orange', alpha=0.2) 
    
    date_in_1months = now_moment + relativedelta(months=1) 
    date_in_2months = now_moment + relativedelta(months=2) 
    date_in_3months = now_moment + relativedelta(months=3) 
    date_in_4months = now_moment + relativedelta(months=4) 
    #ax.axvline(date_in_3months, color='grey', linestyle='--')
 
    
    ax.axvline(date_12months_ago, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_11months_ago, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_10months_ago, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_9months_ago, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_8months_ago, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_7months_ago, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_6months_ago, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_5months_ago, color='grey', linestyle='--',alpha=0.2)    
    ax.axvline(date_4months_ago, color='grey', linestyle='--',alpha=0.2) 
    ax.axvline(date_3months_ago, color='grey', linestyle='--',alpha=0.2) 
    ax.axvline(date_2months_ago, color='grey', linestyle='--',alpha=0.2) 
    ax.axvline(date_1months_ago, color='grey', linestyle='--',alpha=0.2)     
    ax.axvline(end_date, color='grey', linestyle='--')  
    #ax.axvspan(date_6months_ago,end_date, color='orange', alpha=0.2)

    ax.axvline(date_in_1months, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_in_2months, color='grey', linestyle='--',alpha=0.2)
    ax.axvline(date_in_3months, color='grey', linestyle='--')
    ax.axvspan(now_moment,date_in_3months, color='orange', alpha=0.5)    
    
    ax.text(0.3, 0.92, 'past', transform=ax.transAxes,fontsize=20,color='black')
    ax.text(0.85, 0.92, 'prediction', transform=ax.transAxes,fontsize=20,color='black')
    
    ax.set_xlim([date_12months_ago,date_in_3months])
    
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])    
    #ax.set_xticks([])
    #ax.set_yticks([])    
    
    ax.set_ylim([-limit,limit])

    plt.show()
    #fig.savefig('performance.png')

    display(fig)
    
# Visualization of prediction
#visualization_time_series(ts_balance_pd_N, start_date, end_date, N_days_X, N_days_y)    

# COMMAND ----------

# Random selection of N time series of each type:
N = 20
#ts_balance_pd.groupby('signal_type', group_keys=False).apply(pd.DataFrame.sample, n=N).head(50)
ts_balance_pd_N = ts_balance_pd.sample(frac=1).groupby('signal_type').head(N)
ts_balance_pd_N.head(20)

# Visualization of prediction
visualization_time_series(ts_balance_pd_N, start_date, end_date, N_days_X, N_days_y,limit=300000)

# COMMAND ----------

# Random selection of N time series of each type:
N = 20
#ts_balance_pd.groupby('signal_type', group_keys=False).apply(pd.DataFrame.sample, n=N).head(50)
ts_balance_pd_N = ts_balance_pd.sample(frac=1).groupby('signal_type').head(N)
ts_balance_pd_N.head(20)

# Visualization of prediction
visualization_time_series(ts_balance_pd_N, start_date, end_date, N_days_X, N_days_y)

# COMMAND ----------

# Random selection of N time series of each type:
N = 1
#ts_balance_pd.groupby('signal_type', group_keys=False).apply(pd.DataFrame.sample, n=N).head(50)
ts_balance_pd_N_2 = ts_balance_pd.sample(frac=1).groupby('signal_type').head(N)
ts_balance_pd_N_2.head(20)

# Visualization of prediction
visualization_time_series(ts_balance_pd_N_2, start_date, end_date, N_days_X, N_days_y)

# COMMAND ----------

# Random selection of N time series of each type:
N = 1
#ts_balance_pd.groupby('signal_type', group_keys=False).apply(pd.DataFrame.sample, n=N).head(50)
ts_balance_pd_N_3 = ts_balance_pd.sample(frac=1).groupby('signal_type').head(N)
ts_balance_pd_N_3.head(20)

# Visualization of prediction
visualization_time_series(ts_balance_pd_N_3, start_date, end_date, N_days_X, N_days_y)

# COMMAND ----------

# Random selection of N time series of each type:
N = 1
#ts_balance_pd.groupby('signal_type', group_keys=False).apply(pd.DataFrame.sample, n=N).head(50)
ts_balance_pd_N_4 = ts_balance_pd.sample(frac=1).groupby('signal_type').head(N)
ts_balance_pd_N_4.head(20)

# Visualization of prediction
visualization_time_series(ts_balance_pd_N_4, start_date, end_date, N_days_X, N_days_y)

# COMMAND ----------



# COMMAND ----------

ts_balance_pd.head()

# COMMAND ----------

len(ts_balance_pd.balance[0])

# COMMAND ----------

ts_balance_real_pd.head()

# COMMAND ----------

# REAL DATA LOADING
real_data = spark.read.format('parquet').load(cwd+"cashflow_balance_time_series_format_anonymised_000.parquet")
real_data.show(5)
real_data.dtypes

# COMMAND ----------

real_data = real_data.withColumn('balance', real_data.balance.cast("array<float>"))
real_data.show(5)
real_data.dtypes

# COMMAND ----------

real_data = real_data.withColumn('signal_type',F.lit(1))
real_data.show(5)
ts_balance_real_pd = real_data.limit(100).toPandas()

# COMMAND ----------

len(ts_balance_real_pd.balance[0])

# COMMAND ----------

ts_balance_real_pd.head()

# COMMAND ----------

for index, row in ts_balance_real_pd.iterrows():
    print(ts_balance_real_pd.iloc[index]['balance'], row['balance'])    
    if index > 2: break

# COMMAND ----------

# Random selection of N time series of each type:
N = 20
#ts_balance_pd.groupby('signal_type', group_keys=False).apply(pd.DataFrame.sample, n=N).head(50)
ts_balance_pd_N = ts_balance_real_pd.sample(frac=1).groupby('signal_type').head(N)
ts_balance_pd_N.head(20)

# Visualization of prediction
start_date_real='2018-12-01'
end_date_real='2020-03-31'
visualization_time_series(ts_balance_pd_N, start_date_real, end_date_real, N_days_X, N_days_y,limit=100000)

# COMMAND ----------

# We need a tool that tags each time series 0/1 if
# - (option 1) made only of zeroes, or
# - (option 2) made of too little changes, below a threshold defined (threshold_count, for example 10)

real_data.show(5)

# COMMAND ----------

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
  
# trend computation
#ts_balance = ts_balance.withColumn('balance_trend_1MW',
#                                   F.udf(lambda x, y: trend(x,y), "array<float>")('balance', F.lit(30))) 

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
    
real_data = real_data.withColumn('keep_ts',
                                   F.udf(lambda x,y: time_series_cleaning(x,y), "int")('balance', F.lit(10)))     

real_data.show(20)

# COMMAND ----------

real_data.count()
real_data_small = real_data.where('keep_ts == 1')
real_data_small.count()

# COMMAND ----------

real_data.count()

# COMMAND ----------


a = np.array([1,1,1,5,6,6,6])

np.diff(a).astype(bool).sum(axis=0)

# COMMAND ----------

def time_series_cleaning(ts,start_date,end_date,choice_cleaning=2,threshold_count=10,verbose=0):
    '''
    This function removes the time series which are 
    - (option 1) made only of zeroes, or
    - (option 2) made of too little changes, below a threshold defined (threshold_count)
    It is indeed sometimes unuseful (and harmful for convergence) to train autoencoder 
    on time series containing little information
    
    Input  : - ts              : the time series pandas dataframe
             - start_date      : start of train set (format as a datetime python object like: datetime(2017, 1, 1) )
             - end_date        : end of train set (format as a datetime python object like: datetime(2017, 1, 1) )
             - choice_cleaning : the switch for option 1 or 2 (default 2)  
             - threshold_count : the threshold of number of changes below which a time series is removed
             - verbose         : a switch, verbose=0 is mute, verbose=1 prints some statistics
    
    Output : - ts              : the same time series dataframe, but with removed time series,
                                 following option 1 or 2. The output ts dataframe has thus 
                                 reduced row number, compared to input one.
    '''

    if verbose==1: print('Shape of input dataframe:', ts.shape)
    
    if choice_cleaning == 1:
        # We exclude time series made only of zero in the training set
        filter_X = (ts['transactiondate']>=start_date)&(ts['transactiondate']<=end_date)
        train_set = ts.loc[filter_X]
        columns_to_exclude = train_set.loc[:,(train_set == 0).all()].columns
        print('Number of columns with only zeros in train set: ', len(columns_to_exclude))

        # drop by column name
        ts = ts.drop(columns_to_exclude, axis=1)

    if choice_cleaning == 2:
        # Here we exclude time series with less than 10 CHANGES in the time series
        filter_X = (ts['transactiondate']>=start_date)&(ts['transactiondate']<=end_date)
        train_set = ts.loc[filter_X]
        count_non_zeros = []

        # This gives the counts of non-zeros CHANGES (diff) in each ts
        for i,column in enumerate(train_set.columns[1:]):
            if train_set[column].diff().astype(bool).sum(axis=0) > threshold_count:
                count_non_zeros.append(1)
            else:
                count_non_zeros.append(0)
           
        bool_list = list(map(bool,count_non_zeros))
        ts = ts.iloc[:,[True]+bool_list] #first column is the transactiondate
        
        if verbose==1: 
            print('Fraction of columns with more than {} changes: '.format(threshold_count), \
                  sum(count_non_zeros)/len(count_non_zeros) )
            print('Shape of output dataframe:', ts.shape)        
    
    return ts
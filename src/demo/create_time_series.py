
# ===========================
# importing the packages 
# ===========================

import os
import sys, os, inspect
# https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)
#import common.utils as utils
import utils as utils
import json
import pandas as pd
import numpy as np
from pandas import datetime
from dateutil.relativedelta import relativedelta
import random
from scipy import signal
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
import pyspark.sql.functions as F
import pyspark.sql.types as pst
from pyspark.sql import Row

# ===========================
# defining the spark session
# ===========================

spark = SparkSession.builder.getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "false")

# detection of local vs remote (databricks) environment
setting = spark.conf.get("spark.master")
if "local" in setting:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
else:
    print("Do nothing - dbutils should be available already")

print(setting)
print(dbutils.fs.ls("dbfs:/"))
print(dbutils.secrets.listScopes())

# mount point definition (mount has been done in notebook in databricks)
#cwd = "/dbfs/mnt/demo/"
cwd = "/mnt/demo/"

if "local" in setting:
    ROOT_DIR = os.path.dirname(os.path.abspath("README.md"))
    print(ROOT_DIR) 
else:    
    ROOT_DIR = "" 

# df = spark.read.csv("/mnt/demo/sampledata.csv")
# df.show()

# print(spark.conf.get("spark.home")) not working locally
# print(dbutils.fs.ls("dbfs:/libraries/"))


# ===========================
# Reading configuration files
# ===========================

data_conf = utils.Get_Data_From_JSON(ROOT_DIR + "/dbfs" + cwd + "data.json")
model_conf = utils.Get_Data_From_JSON(ROOT_DIR + "/dbfs" + cwd + "config.json")
print(data_conf)
print(model_conf)

# Time window and number of customers
start_date, end_date = data_conf['start_date'], data_conf['end_date']
N_days_X, N_days_y = int(data_conf['number_of_historical_days']), int(data_conf['number_of_predicted_days'])  # 365, 92

end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
start_date_for_prediction_dt = end_date_dt - relativedelta(days=N_days_X + N_days_y)
start_date_for_prediction = start_date_for_prediction_dt.strftime("%Y-%m-%d")

start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay, end_date_minus_6month = utils.dates_definitions(
    start_date, end_date, N_days_X, N_days_y)

time_range = pd.date_range(start_date, end_date, freq='D')

# Type of dataset desired
# Case we want a dataset to train a model: use 1e5 and serving_mode=False
# Case we want an unseen dataset to serve the model on: use 2.5e6 and serving_mode=True
N_customers = 1e2 #2.5e6
serving_mode = False  # True if creating data for serving


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


# ============================
# Generation of synthetic data (for both formats)
# ============================

dff = spark.range(N_customers).toDF("primaryaccountholder")

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


# ========================
# Saving the dataset
# ========================
 
if not serving_mode:
    table_out = data_conf['dev']['table_to_train_on'] 
else:    
    table_out = data_conf['dev']['table_to_score']    

timeseries_spark.write.mode("overwrite").parquet(cwd+"{0}.parquet".format(table_out))
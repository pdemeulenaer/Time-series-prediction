# Databricks notebook source
# Declaration of input parameter: the environment selected
#dbutils.widgets.removeAll()
#dbutils.widgets.dropdown("environment", "test", 
#                         ["test", "prod"], "The environment selected for inference")

# COMMAND ----------

# ---------------------------------------------------------------------------------------------------------------------
# If run in production, in MLflow model registry, the last model currently in "Staging" is transitioned to "Production"
# ---------------------------------------------------------------------------------------------------------------------

from mlflow.tracking.client import MlflowClient

# Define the environment (dev, test or prod)
environment = dbutils.widgets.getArgument("environment")

if environment == 'prod':
  
    # Detect the last model currently in "Staging" in MLflow model registry.    
    mlflow_model_name = 'super_test'
    mlflow_model_stage = 'Staging'

    client = MlflowClient()
    for mv in client.search_model_versions("name='{0}'".format(mlflow_model_name)):
        if dict(mv)['current_stage'] == mlflow_model_stage:
            model_dict = dict(mv)
                        
            print('Model extracted run_id: ', model_dict['run_id'])
            print('Model extracted version number: ', model_dict['version'])
            print('Model extracted stage: ', model_dict['current_stage'])    

            # Transition the registered model stage from "None" to "Staging"            
            client.transition_model_version_stage(
                name=mlflow_model_name,
                version=model_dict['version'],
                stage="Production",
            )   
            
            print()
            print('Model transitioned to Production')
            break  

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# -*- coding: utf-8 -*-

import os
import sys
import traceback
import logging
import logging.config
import yaml
import json
import time
import pandas as pd
from pandas import Series
import numpy as np
import random
import warnings
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# from tensorflow.contrib import predictor
# from tensorflow.contrib.data import unbatch
# from petastorm import make_batch_reader
# from petastorm.tf_utils import make_petastorm_dataset
# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
# from pyspark.sql.window import Window
from pyspark.sql.functions import pandas_udf,PandasUDFType
import pyspark.sql.functions as F
import mlflow
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking.client import MlflowClient

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.shuffle.partitions", "1000")
spark.conf.set("spark.default.parallelism", "1000") # this is valid only for RDDs
spark.conf.set("spark.databricks.io.cache.enabled", "false")

spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "false")

# blob_name = "blob1"
# account_name = "aacdlml0461491171"
# storageKey1 = dbutils.secrets.get(scope = "key-vault-secrets-cloudai", key = "storageaccountcloudaiKey1")
# spark.conf.set("fs.azure.account.key."+account_name+".blob.core.windows.net", storageKey1)

# # For reading from mount point (if not already mounted)
# try: 
#     dbutils.fs.mount(
#         source = "wasbs://"+blob_name+"@"+account_name+".blob.core.windows.net/",
#         mount_point = "/mnt/test",
#         extra_configs = {"fs.azure.account.key."+account_name+".blob.core.windows.net":dbutils.secrets.get(scope = "key-vault-secrets-cloudai", key = "storageaccountcloudaiKey1")})    
# except:
#     pass
# cwd = "/dbfs/mnt/test/"
cwd = "/dbfs/mnt/delta/"

# Reading configuration files
# data_conf = Get_Data_From_JSON(cwd + "data.json")
# model_conf = Get_Data_From_JSON(cwd + "config.json")

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
    "test": {    
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
    "model_name": "super_test",
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

# Define the environment (dev, test or prod)
environment = dbutils.widgets.getArgument("environment")



# # Define the log file
# with open(cwd + 'logging.conf') as f:
#     config = yaml.safe_load(f.read())
# logging.config.dictConfig(config)
# logger = logging.getLogger(cwd + 'cashflow')


# ---------------------------------------------------------------------------------------
# Main SERVING Entry Point
# ---------------------------------------------------------------------------------------

def score(data_conf, model_conf, evaluation=False, **kwargs):

    try:
        print("----------------------------------")
        print("Starting Cashflow DL Model Scoring")
        print("----------------------------------")
        print("")

        # ==============================
        # 0. Main parameters definitions
        # ==============================

        # Size of X and y arrays definition
        N_days_X, N_days_y = int(data_conf['number_of_historical_days']), int(data_conf['number_of_predicted_days']) #365, 92
        print('Number of days used for prediction (X): {0}'.format(N_days_X))
        print('Number of days predicted (y): {0}'.format(N_days_y))
        print('')

        # Date range definition
        start_date, end_date = data_conf['start_date'], data_conf['end_date']
        start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay, end_date_minus_6month = dates_definitions(start_date, end_date, N_days_X, N_days_y)
        print('Date range: [{0}, {1}]'.format(start_date, end_date))
        print('')

        model_name = model_conf['model_name']
    
        #print("Step 0 completed (main parameters definition)")

    except Exception as e:
        print("Errored on initialization")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ==================================
        # S.1 Pre-processings before serving
        # ==================================

        start_time_S1 = time.time()

        # Loading dataset
        table_in = data_conf[environment]['table_to_score']
        
        #ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()
        ts_balance = spark.read.format("delta").load("/mnt/delta/{0}".format(table_in))

        print('Reading table {0}'.format(table_in))
        #print('Size of table: ',ts_balance.count())
        #print('ts_balance.rdd.getNumPartitions()',ts_balance.rdd.getNumPartitions())

        if not evaluation:
            ts_balance = pre_processing(ts_balance, end_date, spark, serving=True)
        if evaluation:
            ts_balance = pre_processing(ts_balance, end_date, spark, serving=False)
        ts_balance.show(3)

        # Saving prepared dataset
        table_out = data_conf[environment]['cashflow_s1_out_scoring']
        
        #ts_balance.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))
        ts_balance.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format(table_out))

        ts_balance.unpersist()
        spark.catalog.clearCache()
        end_time_S1 = time.time()
        print("Step S.1 completed: pre-processings before serving")
        print("Time spent: ", end_time_S1-start_time_S1)

    except Exception as e:
        print("Errored on step S.1: pre-processings before serving")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ===================
        # S.2 Model serving
        # ===================

        start_time_S2 = time.time()

        # Loading dataset
        table_in = data_conf[environment]['cashflow_s1_out_scoring']
        
        #ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in))
        ts_balance = spark.read.format("delta").load("/mnt/delta/{0}".format(table_in))
        ts_balance.cache()
        print('Number of  partitions: ', ts_balance.rdd.getNumPartitions())
        
        # Load model from MLflow model registry #https://www.mlflow.org/docs/latest/model-registry.html        
        mlflow_model_name = model_conf['model_name']
        if environment == 'prod' : 
            mlflow_model_stage = 'Production'
        else:
            mlflow_model_stage = 'Staging'
            
        # Detecting the model dictionary among available models in MLflow model registry. 
        client = MlflowClient()
        for mv in client.search_model_versions("name='{0}'".format(mlflow_model_name)):
            if dict(mv)['current_stage'] == mlflow_model_stage:
                model_dict = dict(mv)
                break  
                
        print('Model extracted run_id: ', model_dict['run_id'])
        print('Model extracted version number: ', model_dict['version'])
        print('Model extracted stage: ', model_dict['current_stage'])                

        def get_local_path_from_dbfs(dbfs_path):
            '''
            This get the local version of the dbfs path, i.e. replaces "dbfs:" by "/dbfs", for local APIs use.
            '''
            #os.path.join("/dbfs", dbfs_path.lstrip("dbfs:"))  #why does not work??? 
            return "/dbfs"+dbfs_path.lstrip("dbfs:")  

        mlflow_path = get_local_path_from_dbfs(model_dict['source']) + '/tfmodel'       
        print("mlflow_path: ",mlflow_path)

        # It detects the name id of the pb model file  
        file = [f for f in os.listdir('/dbfs/mnt/test/{0}/model/'.format(model_name))]
        print(file)
        export_dir_saved = "/dbfs/mnt/test/{0}/model/".format(model_name)#+file[0]   # TODO!!! GET THE MODEL FROM MLFLOW !!!!
        print(export_dir_saved)
        
        #def rdd_scoring(numpy_array):
        #    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = export_dir_saved)
        #    return predictor_fn({'input': numpy_array.reshape(-1, N_days_X, 1) })        
          
        #@F.udf("array<float>")
        #def udf_scoring(x):
        #    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = mlflow_path) #export_dir_saved)
        #    return np.around(predictor_fn({'input': np.array(x).reshape(-1, N_days_X, 1) })['output'][0].tolist(), decimals=3).tolist()                         
        
        @F.pandas_udf("array<float>")
        def pandas_udf_scoring(x):
            #predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = export_dir_saved) #mlflow_path) 
            #return Series([np.around(predictor_fn({'input': np.array(v).reshape(-1, N_days_X, 1)})['output'][0], decimals=3) for v in x])            
            new_model = tf.keras.models.load_model(export_dir_saved)
            #new_model = mlflow.tensorflow.load_model(mlflow_path)
            return Series([np.around( new_model.predict( np.array(v).reshape(-1, N_days_X, 1) ).reshape(N_days_y), decimals=3) for v in x])       

        ts_balance = ts_balance.withColumn('y_pred', pandas_udf_scoring('X'))
        #ts_balance = ts_balance.withColumn('y_pred', udf_scoring('X'))

        print('ts_balance.rdd.getNumPartitions()',ts_balance.rdd.getNumPartitions())
        ts_balance.show(3)
        #print('Size of table: ',ts_balance.count())

        # Saving prepared dataset
        table_out = data_conf[environment]['cashflow_s2_out_scoring']
        
        #ts_balance.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))
        ts_balance.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format(table_out))

        ts_balance.unpersist()
        spark.catalog.clearCache()
        end_time_S2 = time.time()
        print("Step S.2 completed: model serving")
        print("Time spent: ", end_time_S2-start_time_S2)

    except Exception as e:
        print("Errored on step S.2: model serving")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ===================
        # S.3 Post-processing
        # ===================

        start_time_S3 = time.time()

        # Loading dataset
        table_in = data_conf[environment]['cashflow_s2_out_scoring']
        
        #ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()
        ts_balance = spark.read.format("delta").load("/mnt/delta/{0}".format(table_in))

        ts_balance = post_processing(ts_balance)
        ts_balance.show(3)

        # Saving prepared dataset
        table_out = data_conf[environment]['table_scored']
        
        #ts_balance.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))
        ts_balance.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format(table_out))

        ts_balance.unpersist()
        end_time_S3 = time.time()
        print("Step S.3 completed: post-processing")
        print("Time spent: ", end_time_S3-start_time_S3)

    except Exception as e:
        print("Errored on step S.3: post-processing")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e


def evaluate(data_conf, model_conf, scoring=True, **kwargs):

    try:
        print("-------------------------------------")
        print("Starting Cashflow DL Model Evaluation")
        print("-------------------------------------")
        print()

        # ==============================
        # 0. Main parameters definitions
        # ==============================

        # Size of X and y arrays definition
        N_days_X, N_days_y = int(data_conf['number_of_historical_days']), int(data_conf['number_of_predicted_days']) #365, 92
        print('Number of days used for prediction (X): ', N_days_X)
        print('Number of days predicted (y): ', N_days_y)
        print()

        # Date range definition
        start_date, end_date = data_conf['start_date'], data_conf['end_date']
        start_date_dt, end_date_dt, start_date_prediction, end_date_prediction, end_date_plusOneDay, end_date_minus_6month = dates_definitions(start_date, end_date, N_days_X, N_days_y)
        print('Date range: ', start_date, end_date)
        print()

        model_name = model_conf['model_name']                
       

    except Exception as e:
        print("Errored on initialization")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ===========================
        # E.1 Scoring of test data
        # ===========================

        #if kwargs['do_we_score'] is True: # switch, in case we want to skip score (if score already computed earlier)
        if scoring: # switch, in case we want to skip score (if score already computed earlier)
            score(data_conf, model_conf, evaluation=True) # the score function is applied on test dataset for performance evaluation

    except Exception as e:
        print("Errored on step E.1: scoring of test data")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e      

    try:
        # ===========================
        # E.2 Metrics & Visualization
        # ===========================

        # Load model from MLflow model registry #https://www.mlflow.org/docs/latest/model-registry.html        
        #mlflow_model_name = 'cashflow-poc'
        mlflow_model_name = model_conf['model_name']
        if environment == 'prod' : 
            mlflow_model_stage = 'Production'
        else:
            mlflow_model_stage = 'Staging'
            
        # Detecting the model dictionary among available models in MLflow model registry. 
        client = MlflowClient()
        for mv in client.search_model_versions("name='{0}'".format(mlflow_model_name)):
            if dict(mv)['current_stage'] == mlflow_model_stage:
                model_dict = dict(mv)
                break     
                
        print('Model extracted run_id: ', model_dict['run_id'])
        print('Model extracted version number: ', model_dict['version'])
        print('Model extracted stage: ', model_dict['current_stage'])
        
        #MLflow logging of metrics for trained model
        mlflow.end_run() # in case mlfow run_id defined before here
        mlflow.start_run(run_id=model_dict['run_id'])       
        #mlflow.start_run()  # specify the runid!!!
        
        # Loading dataset
        table_in = data_conf[environment]['table_scored']        
        #ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()
        ts_balance = spark.read.format("delta").load("/mnt/delta/{0}".format(table_in))

        # Extracting the test set to Pandas
        ts_balance_pd = ts_balance.select('balance','X', 'y','y_pred','y_pred_rescaled_retrended').toPandas()

        # Extraction of metrics
        R2_all_3month, R2_array_3month, R2_all_1month, R2_array_1month = metric_extraction(ts_balance_pd, N_days_y)

        # Visualization of prediction
        #fig1, fig2 = visualization_prediction(ts_balance_pd, start_date, end_date, N_days_X, N_days_y, R2_array_1month, R2_array_3month, serving=False)
        fig1, fig2 = visualization_time_series_pred_only(ts_balance_pd, start_date, end_date, N_days_X, N_days_y, R2_array_1month, R2_array_3month, serving=False)
        fig1.savefig('/dbfs/mnt/delta/performance.png')   
        fig2.savefig('/dbfs/mnt/delta/performance_R2.png')        
        mlflow.log_artifact('/dbfs/mnt/delta/performance.png') 
        mlflow.log_artifact('/dbfs/mnt/delta/performance_R2.png')        
        
        # Saving the metric
        print('Test R2 metric (3-months window): {}'.format(R2_all_3month))
        print('Test R2 metric (1-months window): {}'.format(R2_all_1month))        
        mlflow.log_metric("R2_all_3month", R2_all_3month)
        mlflow.log_metric("R2_all_1month", R2_all_1month)

        with open("/dbfs/mnt/delta/evaluation.json", "w+") as f:          
            json.dump({'R2_3month': R2_all_3month, 'R2_1month': R2_all_1month}, f)            
        mlflow.log_artifact("/dbfs/mnt/delta/evaluation.json")
        
        mlflow.end_run()

        ts_balance.unpersist()
        print("Step E.2 completed visualisation")

    except Exception as e:
        print("Errored on step E.2: visualisation")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e
        
        
if __name__ == "__main__":
    if dbutils.widgets.getArgument("environment") == 'prod' : # if prod, the model is served
        score(data_conf, model_conf, evaluation=False)
    else:
        evaluate(data_conf, model_conf, scoring=True)   # if NOT prod, the model is evaluated

# COMMAND ----------

# import mlflow
# mlflow.__version__

# COMMAND ----------

# %fs ls dbfs:/mnt/test/super_test/model/

# COMMAND ----------

# %fs ls dbfs:/databricks/mlflow-tracking/1103167146829566/27f38b9005fa43f9990ca8b45a139692/artifacts/model

# COMMAND ----------

# from mlflow.tracking.client import MlflowClient
# import mlflow

# mlflow_model_name = 'super_test'
# mlflow_model_stage = 'Production'
# client = MlflowClient()
# for mv in client.search_model_versions("name='{0}'".format(mlflow_model_name)):
#     if dict(mv)['current_stage'] == mlflow_model_stage:
#         model_dict = dict(mv)
#         break     
        
# print('Model extracted run_id: ', model_dict['run_id'])
# print('Model extracted version number: ', model_dict['version'])
# print('Model extracted stage: ', model_dict['current_stage'])

# model = mlflow.tensorflow.load_model(model_uri="/dbfs/databricks/mlflow-tracking/1103167146829566/27f38b9005fa43f9990ca8b45a139692/artifacts/model")
# model.summary()

# COMMAND ----------

# %fs ls dbfs:/

# COMMAND ----------

# %fs ls dbfs:/databricks/mlflow-tracking

# COMMAND ----------


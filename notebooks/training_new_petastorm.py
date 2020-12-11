# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

# -*- coding: utf-8 -*-

import os
import sys
import traceback
import logging
import logging.config
import yaml
import time as tm
import math
import json
import pandas as pd
import numpy as np
import random
import warnings

import tensorflow as tf
import mlflow
import mlflow.tensorflow
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

mlflow.set_experiment("/Shared/cashflow/cashflow_experiment")

# blob_name = ""
# account_name = ""
# storageKey1 = dbutils.secrets.get(scope = "key-vault-secrets-cloudai", key = "storageaccountcloudaiKey1")
# spark.conf.set("fs.azure.account.key."+account_name+".blob.core.windows.net", storageKey1)

# For reading from mount point (if not already mounted)
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
environment = 'dev' #dbutils.widgets.getArgument("environment")

# Define the log file
# with open(cwd + 'logging.conf') as f:
#     config = yaml.safe_load(f.read())
# logging.config.dictConfig(config)
# logger = logging.getLogger(cwd + 'cashflow')


# ---------------------------------------------------------------------------------------
# Main TRAINING Entry Point
# ---------------------------------------------------------------------------------------
def train(data_conf, model_conf, **kwargs):
  
    try:
        print("-----------------------------------")
        print("Starting Cashflow DL Model Training")
        print("-----------------------------------")
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
        # ========================================
        # T.1 Pre-processing before model training
        # ========================================

        # Loading dataset
        table_in = data_conf[environment]['table_to_train_on']
        #ts_balance = spark.read.parquet("/mnt/test/{0}.parquet".format(table_in)).cache()
        ts_balance = spark.read.format("delta").load("/mnt/delta/{0}".format(table_in))
      
        # Cleaning of the time series
        ts_balance = ts_balance.withColumn('balance', ts_balance.balance.cast("array<float>"))
        
        ts_balance = ts_balance.withColumn('keep_ts', F.udf(lambda x,y: time_series_cleaning(x,y), "int")('balance', F.lit(20)))  #at least 10 transactions in the ts, to be used in the training
        
        ts_balance = ts_balance.where('keep_ts == 1')

        # Creating the dataset on which we train (and test and validate) the model
        ts_balance_model = ts_balance.sample(False, 0.7, seed=0) #now 0.7, but in real case would be 0.1 at best... or 0.05            
        print('ts_balance_model.count()',ts_balance_model.count())

        # Pre-processing before model training
        ts_balance_model = pre_processing(ts_balance_model,
                                          end_date,
                                          spark,
                                          serving=False)
        ts_balance_model.show(3)

        print('ts_balance_model.rdd.getNumPartitions()',ts_balance_model.rdd.getNumPartitions())
        ts_balance_model.show(3)

        # Saving prepared dataset
        table_out = 'cashflow_training_step1'
        #ts_balance_model.write.format("parquet").mode("overwrite").save("/mnt/test/{0}.parquet".format(table_out))
        ts_balance_model.write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format(table_out))

    except Exception as e:
        print("Errored on step T.1: pre-processing before model training")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ========================================
        # T.2 Generating TRAIN, VAL, TEST datasets
        # ========================================

        # Loading datasets
        table_model = 'cashflow_training_step1'
        #ts_balance_model = spark.read.parquet("/mnt/test/{0}.parquet".format(table_model)).cache()
        ts_balance_model = spark.read.format("delta").load("/mnt/delta/{0}".format(table_model)).cache()
        ts_balance_model.show(3)

        print('ts_balance_model.count()', ts_balance_model.count())
        print('ts_balance_model.rdd.getNumPartitions()', ts_balance_model.rdd.getNumPartitions())

        train_set, val_set, test_set = ts_balance_model.randomSplit([0.6, 0.2, 0.2], seed=12345)
        train_set.show(3)
        print('train_set.rdd.getNumPartitions(), val_set.rdd.getNumPartitions(), test_set.rdd.getNumPartitions()',
              train_set.rdd.getNumPartitions(), val_set.rdd.getNumPartitions(), test_set.rdd.getNumPartitions())

        # Saving prepared datasets (train, val, test sets to parquet)
        table_train = 'cashflow_train'
        table_val = 'cashflow_val'
        table_test = data_conf[environment]['table_test_for_performance'] #'cashflow_test'

        train_set.select('X','y').write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format(table_train))
        val_set.select('X','y').write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format(table_val))
        test_set.select('primaryaccountholder','transactiondate','balance')\
            .write.format("delta").mode("overwrite").save("/mnt/delta/{0}".format(table_test))

    except Exception as e:
        print("Errored on step T.2: pre-processings")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e

    try:
        # ==============================
        # T.3 MODEL DEFINITION AND TRAIN
        # ==============================
        
        table_train = 'cashflow_train'
        table_val = 'cashflow_val'
        #table_train = spark.read.parquet("/mnt/test/{0}.parquet".format(table_train))
        table_train = spark.read.format("delta").load("/mnt/delta/{0}".format(table_train))
        #table_val = spark.read.parquet("/mnt/test/{0}.parquet".format(table_val))
        table_val = spark.read.format("delta").load("/mnt/delta/{0}".format(table_val))
        table_train_count = table_train.count()
        table_val_count = table_val.count()
        #table_train_count, table_val_count            
        
        from pyspark.sql.functions import col
        from petastorm.spark import SparkDatasetConverter, make_spark_converter

        # Set a cache directory on DBFS FUSE for intermediate data.
        spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")
        converter_train = make_spark_converter(table_train)
        converter_val = make_spark_converter(table_val)

        print(f"train: {len(converter_train)}, val: {len(converter_val)}") 
        
        def get_compiled_model(N_days_X, N_days_y, model_conf): #lr=0.001
            #model = get_model(lr=lr)
            model = define_1dcnn_model(N_days_X, N_days_y, model_conf)

            hyperparameters = model_conf['hyperParameters']

            opt = tf.keras.optimizers.Adam()

            # Model compilation
            model.compile(optimizer=opt, loss=hyperparameters['loss'])  

            return model    
          
        # Enable auto-logging to MLflow to capture TensorBoard metrics.
        mlflow.tensorflow.autolog(every_n_iter=1)

        model_name = model_conf['model_name']
        mlflow_model_name = model_name
        model_dir = "/tmp/"+model_name
        try:
            dbutils.fs.rm(model_dir, recurse=True)
        except OSError:
            pass

        with mlflow.start_run():

            NUM_EPOCHS = model_conf['hyperParameters']['epochs'] #5
            BATCH_SIZE = model_conf['hyperParameters']['batch_size'] #500

            def train_and_evaluate(N_days_X, N_days_y, model_conf): #lr=0.001
                model = get_compiled_model(N_days_X, N_days_y, model_conf) #lr

                with converter_train.make_tf_dataset(batch_size=BATCH_SIZE) as train_dataset, \
                     converter_val.make_tf_dataset(batch_size=BATCH_SIZE) as val_dataset:

                    #train_dataset = train_dataset.map(lambda x: (x.features, x.label_index))
                    train_dataset = train_dataset.map(lambda x: (tf.reshape(x.X, [-1,N_days_X,1]), tf.reshape(x.y, [-1,N_days_y])))       
                    steps_per_epoch = len(converter_train) // BATCH_SIZE

                    #val_dataset = val_dataset.map(lambda x: (x.features, x.label_index))
                    val_dataset = val_dataset.map(lambda x: (tf.reshape(x.X, [-1,N_days_X,1]), tf.reshape(x.y, [-1,N_days_y])))
                    validation_steps = max(1, len(converter_val) // BATCH_SIZE)

                    print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")

                    hist = model.fit(train_dataset, 
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=NUM_EPOCHS,
                                     validation_data=val_dataset,
                                     validation_steps=validation_steps,
                                     verbose=2)
                    return model,hist 

            model, hist = train_and_evaluate(N_days_X, N_days_y, model_conf)
            print(hist.history['val_loss'][-1])   

            #MLflow logging
            #mlflow.log_artifact(cwd + "data.json")
            #mlflow.log_artifact(cwd + "config.json")              
            mlflow.log_param("model_name", str(model_name))        
            mlflow.log_param("N_days_X", N_days_X)
            mlflow.log_param("N_days_y", N_days_y)
            mlflow.log_param("start_date", start_date)
            mlflow.log_param("end_date", end_date)             
            mlflow.log_param("num_epochs", str(NUM_EPOCHS))
            mlflow.log_param("batch_size", str(BATCH_SIZE))
            #mlflow.log_param("steps_per_epoch", str(steps_per_epoch)) #validation_steps 

            # saving using tf.keras.models.save_model
            tf.keras.models.save_model(model,filepath=model_dir+'/model') #SavedModel format
            #model.save(filepath=model_dir+'model', save_format="h5")      #H5 format (todo, and look how to register that)

            # saving using mlflow.tensorflow.save_model (this does NOT log nor register the model) does not overwrites...
            #mlflow.tensorflow.save_model(tf_saved_model_dir=model_dir+'/model',
            #                             tf_meta_graph_tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
            #                             tf_signature_def_key='serving_default',
            #                             path = 'model')

            # logging already saved model
            mlflow.tensorflow.log_model(tf_saved_model_dir=model_dir+'/model',
                                        tf_meta_graph_tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
                                        tf_signature_def_key='serving_default',
                                        registered_model_name=model_name,
                                        artifact_path='model')  
            
            # Getting the version number of the newly registered MLflow model (useful for next steps)
            mlflow_model_version = 0
            client_current_model = MlflowClient()
            for mv in client_current_model.search_model_versions("name='{0}'".format(mlflow_model_name)):
                #if int(dict(mv)['version']) == mlflow_model_version: 
                if int(dict(mv)['version']) >= mlflow_model_version: # finding the last version registered
                    mlflow_model_version = int(dict(mv)['version'])
                    model_dict = dict(mv)  

            #update 2020-07017: to grab the latest model version, we can also do like this: (TO BE TESTED!!!)
            #model_version_infos = client_current_model.search_model_versions(f"name = '{model_name}'")
            #mlflow_model_version = max([model_version_info.version for model_version_info in model_version_infos])                
        
                  
            # Wait until the model is ready
            def wait_until_model_ready(model_name, model_version):
              client = MlflowClient()
              for _ in range(20):
                model_version_details = client.get_model_version(
                  name=model_name,
                  version=model_version,
                )
                status = ModelVersionStatus.from_string(model_version_details.status)
                print("Model status: %s" % ModelVersionStatus.to_string(status))
                if status == ModelVersionStatus.READY:
                  break
                tm.sleep(5)

            wait_until_model_ready(mlflow_model_name, mlflow_model_version)            

            # Transition the registered model stage from "None" to "Staging"            
            client_current_model.transition_model_version_stage(
                name=mlflow_model_name,
                version=mlflow_model_version,
                stage="Staging",
            )    
            
            # Copy the file from the driver node and save it to DBFS (so that they can be accessed e.g. after the current cluster terminates.):
            dbutils.fs.cp("file:/tmp/{0}/model".format(model_name), "dbfs:/mnt/test/{0}/model".format(model_name), recurse=True)
            print('Model copied here: ', "dbfs:/mnt/test/{0}/model/".format(model_name))                   
        
          #mlflow.end_run()

    except Exception as e:
        print("Errored on step T.3: model definition and train")
        print("Exception Trace: {0}".format(e))
        print(traceback.format_exc())
        raise e                          

if __name__ == "__main__":
    train(data_conf, model_conf)

# COMMAND ----------

#%fs ls /mnt/delta/

# COMMAND ----------

#model_name = 'super_test'
#dbutils.fs.cp("file:/tmp/{0}/model".format(model_name), "dbfs:/mnt/test/{0}/model".format(model_name), recurse=True)
#dbutils.fs.rm("/foobar/baz.txt")

# COMMAND ----------

#%fs ls file:/tmp/super_test    

# COMMAND ----------

# %fs ls dbfs:/tmp/

# COMMAND ----------

#display(dbutils.fs.ls("dbfs:/mnt/test/super_test/model/variables"))

# COMMAND ----------

# dbutils.fs.cp("dbfs:/mnt/test/{0}/model".format(model_name), "file:/tmp/{0}/model".format(model_name), recurse=True)
# new_model = tf.keras.models.load_model("file:/tmp/{0}/model".format(model_name))
# new_model.summary()


# COMMAND ----------

# %fs ls file:/tmp/super_test/model

# COMMAND ----------

# dbutils.fs.cp("dbfs:/mnt/test/{0}/model".format(model_name), "file:/tmp/{0}/model".format(model_name), recurse=True)
# new_model = tf.keras.models.load_model("/tmp/super_test/model")
# #new_model = tf.saved_model.load("/tmp/super_test/model")

# new_model.summary()

# COMMAND ----------


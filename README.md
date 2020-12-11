[![Build Status](https://dev.azure.com/pdemeulenaer/test/_apis/build/status/pdemeulenaer.Time-series-prediction?branchName=master)](https://dev.azure.com/pdemeulenaer/test/_build/latest?definitionId=6&branchName=master)

# Time series prediction using 1-D Convolutional Neural Network for big data

The model is described in the associated Medium post: [https://medium.com/p/168b47e54d54](https://medium.com/p/168b47e54d54)


## Problem

The problem is divided into:

* Generate 2 datasets: one that will serve to train the model (100K time series), the other as unseen data (2.5M time series) on which to serve the model.

* Train a 1-D CNN model on `train` dataset

* Serve the model on `unseen` dataset (when serving mode) or on test set (when evaluation mode)


## Files

- `notebooks/data.json`: configuration file for the data generation (temporarily included in the training and scoring codes)

- `notebooks/config.json`: model configuration file (temporarily included in the training and scoring codes)

- `notebooks/synthetic_data_generation.py` : the code for synthetic data generation

- `notebooks/training.py`: the code for model training

- `notebooks/scoring.py`: the code for model serving / evaluation

- `notebooks/utils.py`: the python module containing helper functions

old-autoencoder folder (old): This folder contains examples of how to perform time series forecast using LSTM autoencoders and 1-d convolutional neural networks in Keras

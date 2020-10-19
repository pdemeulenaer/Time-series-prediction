# Time series prediction using 1-D Convolutional Neural Network for big data

The model is described in the associated Medium post: [https://medium.com/p/168b47e54d54](https://medium.com/p/168b47e54d54)


## Problem

The problem is divided into:

* Generate 2 datasets: one that will serve to train the model (100K time series), the other as unseen data (2.5M time series) on which to serve the model.

* Train a 1-D CNN model on `train` dataset

* Serve the model on `unseen` dataset (when serving mode) or on test set (when evaluation mode)


## Files

- `data.json`: configuration file for the data generation (temporarily included in the training and scoring codes)

- `config.json`: model configuration file (temporarily included in the training and scoring codes)

- `synthetic_data_generation.py` : the code for synthetic data generation

- `training.py`: the code for model training

- `scoring.py`: the code for model serving / evaluation

- `utils.py`: the python module containing helper functions

autoencoder folder (old): This folder contains examples of how to perform time series forecast using LSTM autoencoders and 1-d convolutional neural networks in Keras

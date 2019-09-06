# Time-series-prediction
This repo contains examples of how to perform time series forecast using LSTM autoencoders and 1-d convolutional neural networks in Keras

Documentation on LSTM autoencoders for time series prediction
---------------------------------------------------------------------------

This is a toy model for testing the performance of a lstm encoder-decoder scheme (also called lstm autoencoder) on time series forecasting. This work is heavily inspired by the papers:

- Laptev et al. 2017: http://www.roseyu.com/time-series-workshop/submissions/TSW2017_paper_3.pdf   

- Srivastava, Mansimov, Slakhutdinov (2015) https://arxiv.org/abs/1502.04681 

For the implementation, it is largely based on Jason Brownlee's post:

- https://machinelearningmastery.com/lstm-autoencoders/ 

Here are some other relevant posts that inspired this work:

Autoencoders:

- https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjQg5Lvu-DfAhXJ8ywKHdbJBiwQFjABegQICRAC&url=http%3A%2F%2Froseyu.com%2Ftime-series-workshop%2Fsubmissions%2FTSW2017_paper_3.pdf&usg=AOvVaw1DjpBuUh-KrFZzQ0SoAC7o (Laptev conference paper)

- https://machinelearningmastery.com/lstm-model-architecture-for-rare-event-time-series-forecasting/ (Discussion of Laptev paper by Jason Brownlee)

- https://machinelearningmastery.com/lstm-autoencoders/ (general intro, very good. I am using the “composite architecture”)

- https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/ 

- https://blog.keras.io/building-autoencoders-in-keras.html 

- https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/ (Dropout in LSTM)

Statsmodels seasonal decomposition: 

- https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html#statsmodels.tsa.seasonal.seasonal_decompose , http://www.statsmodels.org/stable/release/version0.6.html?highlight=seasonal#seasonal-decomposition 

Documentation on 1D CNN for time series prediction
------------------------------------------------------------------

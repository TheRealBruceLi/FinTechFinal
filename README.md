# FinTechFinal
Final Group Project for CMU 70-339 FinTech by Bruce Li, Evan Yukevich, and Preston Vander Vos

We trained a linear regression and a neural network model to predict cryptocurrency prices based on some objective underlying fundamentals of cryptocurrencies.

Since we were not interested in predicting time sensitive prices, time data points were not included as fundamentals of the cryptocurrency.

The loss functions used in both models was mean squared of the error. The neural network architecture was a forward feed with five layers. The input layer was linear followed by hidden layers consisting of a ReLU then Sigmoid then ReLU again, and finally a linear output layer. The hidden layers were all in the first dimension. Both models were trained for fifteen thousand epochs. The learning rate for the neural net was 5e^-6. The learning rate for the linear regression was 0.01. The validation data was randomly selected and consisted of a third of the overall data gathered. 

The six inputs that we used to predict cryptocurrency prices are as follows: transaction volume, transaction count, market cap, exchange volume, number of active addresses, and median transaction value. 

Neural Network:
  
  Epoch 0:
    
    Training Loss: 61827
    
    Validating Loss: 32456
  
  Epoch 7500:
    
    Training Loss: 4394
    
    Validating Loss: 2476
  
  Epoch 15000:
    
    Training Loss: 3057
    
    Validating Loss: 1824


Linear Regression:
  
  Epoch 0:
    
    Training Loss: 46233
    
    Validating Loss: 46233
  
  Epoch 7500:
    
    Training Loss: 3204
    
    Validating Loss: 3204
  
  Epoch 15000:
    
    Training Loss: 3204
    
    Validating Loss: 3204

Interpretation of Models:
We determined which inputs were important for the neural network based off of how the network performed when the input was held constant. By holding this constant, we essentially took out the input to measure how important it is in predicting cryptocurrency prices. By this analysis, the most important input that we found for prediction was the number of active addresses. When this input was held constant, loss diminished by only two over the course of 15,000 epochs. 

We determined which inputs were important for the linear regression based off of the absolute value of the coefficient, corresponding to each input. This analysis led to market cap being the most important input for the regression. In the neural network, the second most important aspect was the market cap. This is consistent because we can see that in the neural network, loss increased by the second highest when market cap was held constant. 

Steps needed to follow to reproduce our results are as follows: 
Install pytorch and tensorflow for python3
Download code and data from github repository (to same directory)
Run NN.py and LinReg.py for the respective models 


Errors and Improvements:
The current implementation suffers from a few flaws. One such flaw is with the distribution of the data. The current cryptocurrency price data, after normalization, contains most values close together around 0 but also contains very prominent outliers. These outliers are due to the massive spike in cryptocurrency prices at the end of 2017. Before normalizing the data, we should have taken the logs of the values to diminish the impact of the outliers. 

Another issue that we needed to address was the fact that one of our inputs, market cap, was determined by the price of the cryptocurrency that we were trying to estimate. Market cap is calculated by multiplying the supply of the cryptocurrency with the price. Since some coins do not change the supply, the market cap in this case would be able to exactly predict price. Another input that also faced this challenge was the median transaction value. 

Once we took out the market cap from the linear regression, the best indicator for the price was the number of active addresses. Since we believe that active address isn’t as tainted as market cap, we can conclude that active address could be an indicator of the price, especially since it was impactful to both the neural network and linear regression.

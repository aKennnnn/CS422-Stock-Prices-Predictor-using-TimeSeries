# CS422-Stock-Prices-Predictor-using-TimeSeries
Stock Prices Predictor using TimeSeries

I. Introduction
Stock market data is rich in granular details, and predicting future stock prices is a complex task. A time series analysis of the event occurrences over a period of time is a good way to identify patterns and predict future occurrences. This project aims to develop a machine learning project that uses time series forecasting to predict future stock prices based on historical data.

II. Problem Statement
The stock market is a dynamic entity, and predicting stock prices accurately is a challenging task. Investors and traders often rely on various tools and techniques to make reasonable decisions while investing. Our project aims to develop a machine learning model that can predict the future price of a stock based on historical data using time series forecasting. Through the use of regression methods and feature selection for instance, we can gain predictions about how stock prices will end up at. Allowing another tool in making those decisions regarding investments made or not.

III. Machine Learning Methods
We will be using a few machine-learning methods to compare the data in our implementation. Firstly, a times series model used to capture linear dependencies along a timeline, named SARIMA. That also uses seasonal patterns in conjunction with the data to better plot predictions. Something that may be very useful in predicting stocks and their recurring patterns. LSTM, a recurrent neural network, that can capture dependencies over the long-term. Lastly, a model named Random Forest, being an ensemble learning method used for regression tasks. Handling non-linear relationships between the features. We believe using these three to compare and contrast the set of market data will help us to make better predictions.

IV. Data Description
We will be using this dataset: https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data?resource=download
It contains 112,457 samples, along with eight features, the market index, open, high, low, close price for the day, along with the volume of shares traded within that day. Over a span of around 60 years of data from many trading indexes around the world.

V. Description of Data settings or preprocessing
The main preprocessing conducted will include handling missing data, outliers, and converting the data into a suitable format for analysis. Missing data mostly being days the market is closed on holidays and such, will be all null values will be removed. For a better format, both the index values and day values into more suitable label values to be used in the three machine learning methods. We will also remove the volume feature, as it seems our data only starts measuring the volume of trades 37% into the dataset. Meaning it will not be useful to include within our process and skew our predictions.

VI. Experiment Design
Data Collection and Preprocessing - 
Collect historical stock market data for a selected set of companies from reliable sources such as kaggle.com.
Split the dataset by using k-fold, with a value of 10 splits.

Feature Engineering - 
Perform feature scaling and normalization if necessary.
Use feature selection techniques (e.g., Regression) to identify the most important features for predicting stock prices.

 Model Implementation and Evaluation
Implement the SARIMA, LSTM, and Random Forest models using appropriate libraries and frameworks.
Train each model using the training dataset and tune hyperparameters for optimal performance.
Evaluate the models' performance on the testing dataset using appropriate evaluation metrics, such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

Model Comparison - 
Compare the performance of the three models based on the evaluation metrics.
Analyze the strengths and weaknesses of each model in the context of stock price prediction.

Visualization and Interpretation - 
Visualize the predicted stock prices and compare them to the actual stock prices.
Interpret the results and draw conclusions about the effectiveness of the models in predicting stock prices.

Documentation and Reporting - 
Document the entire process, including data collection, preprocessing, feature engineering, model implementation, and evaluation.

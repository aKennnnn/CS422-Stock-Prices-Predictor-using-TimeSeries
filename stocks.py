import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator

# Read Data
data = pd.read_csv("indexProcessed.csv",parse_dates=['Date'])
#print(data.head())

# Separate data by stock market index.
HSI = data.loc[data.Index == 'HSI', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
NYA  = data.loc[data.Index == 'NYA', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
IXIC = data.loc[data.Index == 'IXIC', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
SSE  = data.loc[data.Index == '000001.SS', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]  # SSE Composite Index (000001.SS) 
N225 = data.loc[data.Index == 'N225', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']] 
N100 = data.loc[data.Index == 'N100', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
SZI  = data.loc[data.Index == '399001.SZ', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]  # Shenzhen Index (399001.SZ)
GSPTSE = data.loc[data.Index == 'GSPTSE', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
NSEI = data.loc[data.Index == 'NSEI', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
GDAXI  = data.loc[data.Index == 'GDAXI', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
SSMI = data.loc[data.Index == 'SSMI', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
TWII = data.loc[data.Index == 'TWII', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]
ASI = data.loc[data.Index == 'J203.JO', ['Date','Open','High','Low','Close','Adj Close','CloseUSD']]     # All Share Index (^J203.JO)

# Set indexed by date
HSI.set_index('Date',inplace=True)
NYA.set_index('Date',inplace=True)
IXIC.set_index('Date',inplace=True)
SSE.set_index('Date',inplace=True)
N225.set_index('Date',inplace=True)
N100.set_index('Date',inplace=True)
SZI.set_index('Date',inplace=True)
GSPTSE.set_index('Date',inplace=True)
NSEI.set_index('Date',inplace=True)
GDAXI.set_index('Date',inplace=True)
SSMI.set_index('Date',inplace=True)
TWII.set_index('Date',inplace=True)
ASI.set_index('Date',inplace=True)

# LSTM model
# LSTM stands for long short-term memory networks.
# That are capable of learning long-term dependencies, especially in sequence prediction problems.
def createModel():
    model = Sequential()
    #  Unit
    #  return_sequences -- Whether to return the last output in the output sequence, or the full sequence.
    #  input_shape -- Number of steps with 1 character at a time.
    model.add(LSTM(128,return_sequences=True, input_shape=(steps, 1)))
    model.add(LSTM(64,return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

'''
# Show daily close price in graph
for i in data['Index'].unique():
    plt.plot(data[data['Index'] == i]['Close'])
    plt.title(f'{i} Daily Close Price')
    plt.show()
'''
# Show daily difference between open and close
openNclose = data.sort_values(['Index','Date']).set_index('Date')
openNclose['Open_Close'] = data['Close'].values - data['Open'].values

'''
plt.title('Stocks Daily Price Difference Between Open and Close')
plt.plot(openNclose['Open_Close'])
plt.show()
'''
# contral varables
<<<<<<< HEAD
batchSize = 40      # batch_size. The number of samples that will be propagated through the network.
nEpochs = 10        # number of epochs. 
steps = 30      # Time steps in days. means how many values exist in a sequence. 
=======
batchSize = 40      # batch_size
nEpochs = 10        # number of epochs
steps = 30      # days
>>>>>>> ddda3608b9e65fec564428bceccd6541e00b69d9

# Train and show graphs.
for i in openNclose['Index'].unique():
    fig , axis = plt.subplots(nrows=3 , ncols=1 , figsize=(15,10), constrained_layout = True)
    # Set plots in full screen mode
    manager  = plt.get_current_fig_manager()
    manager .window.state('zoomed')
    temp = openNclose[openNclose['Index'] == i]     # set index of each stocks
    length = len(temp[temp.index > '2020-01-01'])   # number of days after 2020
    trainSet = temp[['Close']][ : -length]
    testSet = temp[['Close']][-length :]
    scaler = StandardScaler()                       # Standardize features by removing the mean and scaling to unit variance.
    # combination of fit and transfor function. Iis used to fit the data into a model and transform it into a form that is more suitable for the model in a single step.
    train = scaler.fit_transform(trainSet.values)   
    test = scaler.transform(testSet.values)
    #  Takes in a sequence of data-points gathered at equal intervals, along with time series parameters such as length of history to produce batches for training/validation.
    timeSeriesTrain = TimeseriesGenerator(train, train, length=steps)
    timeSeriesTest = TimeseriesGenerator(test, test, length=steps)
    model = createModel()   # Input Data Generation for LSTM
    history = model.fit(timeSeriesTrain , batch_size=batchSize , epochs=nEpochs)
    predicts = model.predict(timeSeriesTest)
    index = temp[temp.index > '2020-01-01'].index    
    
    axis[0].plot(openNclose[openNclose['Index'] == i]['Close'])
    axis[0].set_title(f'{i} Daily Close Price')
    axis[1].plot(testSet)
    axis[1].set_title(f'{i} After 2020')
    axis[2].plot(index , scaler.inverse_transform(test) , label='Real')
    axis[2].plot(index[steps : ] , scaler.inverse_transform(predicts) , label='Prediction')
    axis[2].set_title(f'{i} Daily Close Real vs Prediction')
    axis[2].legend(loc="upper left")
    
    #plt.plot(history.history['loss'])
    #plt.title('Loss')
    plt.show()


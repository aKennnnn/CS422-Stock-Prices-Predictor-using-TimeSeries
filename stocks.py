import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# Show daily close price in graph
for i in data['Index'].unique():
    plt.plot(data[data['Index'] == i]['Close'])
    plt.title(f'{i} Daily Close Price')
    plt.show()

# Show daily difference between open and close
openNclose = data.sort_values(['Index','Date']).set_index('Date')
openNclose['Open_Close'] = data['Close'].values - data['Open'].values
print(openNclose)
plt.title('Stocks Daily Price Difference Between Open and Close')
plt.plot(openNclose['Open_Close'])
plt.show()


import warnings as warn
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
data = pd.read_csv('indexProcessed.csv')

data['Date']=pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index(['Date'], inplace=True)

HSI = data.loc[data['Index'] == 'HSI']

HSI['CloseUSD'].plot()
plt.ylabel('Close Price')
plt.xlabel('Date')
plt.show()

# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 1)

# Define the p parameters to take any value between 0 and 3
p = range(0, 3)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# split data
train_data = HSI[:'2019-01-01']['CloseUSD']
test_data = HSI['2019-01-02':]['CloseUSD']

# calculate para
AIC_list = []
SARIMAX_model_list = []

warn.filterwarnings("ignore")

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(train_data, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False,)
            results = mod.fit(disp=0)
            # print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC_list.append(results.aic)
            SARIMAX_model_list.append([param, param_seasonal])
        except:
            continue
print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC_list), SARIMAX_model_list[AIC_list.index(min(AIC_list))][0],SARIMAX_model_list[AIC_list.index(min(AIC_list))][1]))

# select the best model
mod = SARIMAX(train_data,
                order=SARIMAX_model_list[AIC_list.index(min(AIC_list))][0],
                seasonal_order=SARIMAX_model_list[AIC_list.index(min(AIC_list))][1],
                enforce_stationarity=False,
                enforce_invertibility=False)
results = mod.fit(disp=0)

# show diagnostics
results.plot_diagnostics(figsize=(20, 14))
plt.show()

########################################## make pred part ##########################################
pred0 = results.get_prediction(start=len(train_data)+1, end=len(HSI), dynamic=False)
pred0_ci = pred0.conf_int()
pred0_values = pred0.predicted_mean.values

pred1 = results.get_prediction(start=len(train_data)+1, end=len(HSI), dynamic=True)
pred1_values = pred1.predicted_mean.values

# Forecast the next observation
pred2 = results.forecast(steps=len(test_data))
pred2_values = pred2.values

predictions_series = []
predictions_series = pred0_values
Pred_0_Array = np.full(len(train_data), np.nan)
complete_predictions_0 = np.concatenate((Pred_0_Array, pred0_values))
predictions_series_0 = pd.Series(complete_predictions_0, index=HSI.index)

predictions_series = pred1_values
Pred_1_Array = np.full(len(train_data), np.nan)
complete_predictions_1 = np.concatenate((Pred_1_Array, pred1_values))
predictions_series_1 = pd.Series(complete_predictions_1, index=HSI.index)

predictions_series = pred2_values
Pred_2_Array = np.full(len(train_data), np.nan)
complete_predictions_2 = np.concatenate((Pred_2_Array, pred2_values))
predictions_series_2 = pd.Series(complete_predictions_2, index=HSI.index)

# show graph
ax = HSI['CloseUSD'].plot(figsize=(20, 16))
predictions_series_0.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
predictions_series_1.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
ax.fill_between(test_data.index, pred0_ci.iloc[:, 0], pred0_ci.iloc[:, 1], color='r', alpha=.5)
plt.ylabel('Close')
plt.xlabel('Date')
plt.legend()
plt.show()
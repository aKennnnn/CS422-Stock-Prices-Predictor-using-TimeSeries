import warnings as warn
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv('indexProcessed.csv')

data['Date']=pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index(['Date'], inplace=True)

HSI = data.loc[data['Index'] == 'HSI']
HSI = HSI['2010-01-01':]

NYA = data.loc[data['Index'] == 'NYA']
NYA = NYA['2010-01-01':]

def ARIMA_model(history, test_data, N_test_observations):
    model_predictions = []
    for time_point in range(N_test_observations):
        print('time_point - {}'.format(time_point), end='\r')
        model = ARIMA(history, order=(1,0,0))
        model_fit = model.fit()
        output = model_fit.forecast(step=1)
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    return model_predictions

def SARIMA_model(train_data, test_data, order_para, seasonal_para):
    pred = []
    for i in range(len(test_data)):
        print('time_point - {}'.format(i), end='\r')
        mod = SARIMAX(train_data,
                            order=order_para,
                            seasonal_order=seasonal_para,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
        results = mod.fit(disp=0)
        pred.append(results.forecast())
        train_data.append(test_data[i])
    return pred

def get_Para(train_data):
    # Define the d and q parameters to take any value between 0 and 1
    q = d = range(0, 1)

    # Define the p parameters to take any value between 0 and 3
    p = range(0, 3)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

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
            
    return SARIMAX_model_list[AIC_list.index(min(AIC_list))][0], SARIMAX_model_list[AIC_list.index(min(AIC_list))][1]

def disp_graph(pred, test_data, dataset):
    temp_pred = pred
    pred = pred[1:]
    pred = pd.Series(pred)
    pred.index = test_data[1:].index
    fig = plt.figure(figsize=(26, 8))
    pred = pred.loc['2019-01-01':]
    dataset = dataset.loc['2019-01-01':]
    plt.plot(pred, color='blue', marker='o', markersize=5, linestyle='', label='Predict')
    plt.plot(dataset, color='orange', linewidth=2, label='Close Price')
    plt.title('Closing Price Predictions: Actual vs. Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    mse = np.mean((test_data.values - temp_pred)**2)
    mae = np.mean(np.abs(test_data.values - temp_pred))

    print("MSE:", mse)
    print("MAE:", mae)
    return

HSI_train_data = HSI[:'2020-01-01']['CloseUSD']
HSI_test_data = HSI['2020-01-02':]['CloseUSD']
NYA_train_data = NYA[:'2020-01-01']['CloseUSD']
NYA_test_data = NYA['2020-01-02':]['CloseUSD']

HSI_history = [x for x in HSI_train_data]
HSI_model_predictions = []
HSI_N_test_observations = len(HSI_test_data)

NYA_history = [x for x in NYA_train_data]
NYA_model_predictions = []
NYA_N_test_observations = len(NYA_test_data)

HSI_ARIMA_model_predictions = ARIMA_model(HSI_history, HSI_test_data, HSI_N_test_observations)
NYA_ARIMA_model_predictions = ARIMA_model(NYA_history, NYA_test_data, NYA_N_test_observations)

HSI_para_pdq, HSI_para_seasonal= get_Para(HSI_train_data)
NYA_para_pdq, NYA_para_seasonal= get_Para(NYA_train_data)

HSI_SARIMA_model_predictions = SARIMA_model(HSI_history, HSI_test_data, HSI_para_pdq, HSI_para_seasonal)
NYA_SARIMA_model_predictions = SARIMA_model(NYA_history, NYA_test_data, NYA_para_pdq, NYA_para_seasonal)

NYA_pred = NYA_SARIMA_model_predictions
HSI_pred = HSI_SARIMA_model_predictions

disp_graph(NYA_pred, NYA_test_data, NYA['CloseUSD'])
disp_graph(HSI_pred, HSI_test_data, HSI['CloseUSD'])
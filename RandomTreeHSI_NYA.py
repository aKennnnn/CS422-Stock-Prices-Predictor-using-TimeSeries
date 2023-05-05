import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from statistics import mode
import matplotlib.pyplot as plt
# from stocks-converted.py
# Read Data
data = pd.read_csv("indexProcessed.csv", parse_dates=['Date'])
# Convert Data into Numerical Values
data['Year'] = data["Date"].dt.year
data['Month'] = data["Date"].dt.month
data['Day'] = data["Date"].dt.day
# Drop Unnecessary Columns
data = data.drop(['Volume', 'Date'], axis=1)

# Convert Labels into Numerical Values
# 000001.SS = 0
# 399001.SZ = 1
# GDAXI = 2
# GSPTSE = 3
# HSI = 4
# IXIC = 5
# J203.JO = 6
# N100 = 7
# N225 = 8
# NSEI = 9
# NYA = 10
# SSMI = 11
# TWII = 12
lecoder = LabelEncoder()
label = lecoder.fit_transform(data['Index'])
data["Index"] = label

# Set Index Column as Index
data.set_index('Index', inplace=True)

count = 0
accuracies = []
preds = []
while (count < 10):
    dataTest = data.loc[[4]]

    # Add a binary variable indicating whether the stock increased in price or not
    dataTest['price_increase'] = (dataTest['Close'] > dataTest['Open']).astype(int)

    # Separate the input features (X) from the target variable (y)
    y = dataTest['price_increase']
    X = dataTest[['Open', 'High', 'Low', 'Close', 'Adj Close', 'CloseUSD', 'Year', 'Month', 'Day']]


    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a random forest classifier
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Index HSI')
    #print('Test accuracy:', test_accuracy)
    accuracies.append(test_accuracy)
    # Make predictions for a new stock
    new_stock = pd.DataFrame([[100, 110, 90, 95, 100, 9500, 2030, 1+count, 3]], columns=X.columns)
    prediction = rf_model.predict(new_stock)
    listpred = prediction.tolist()
    preds.append(listpred[0])
    print('Prediction:', prediction)
    count = count + 1
print('Most common prediction ', mode(preds))

#graph accuracy
x = range(1, 11)
plt.plot(x, accuracies, 'o-')
plt.xlabel('Test Number')
plt.ylabel('Accuracy')
plt.title('Accuracy for HSI')
plt.show()

#bar graph
plt.bar(x, preds)
plt.xlabel('Test Number')
plt.ylabel('Prediction')
plt.title('HSI')
plt.show()
preds

count = 0
NYAaccuracies = []
NYApreds = []
while (count < 10):
    dataTest = data.loc[[10]]

    # Add a binary variable indicating whether the stock increased in price or not
    dataTest['price_increase'] = (dataTest['Close'] > dataTest['Open']).astype(int)

    # Separate the input features (X) from the target variable (y)
    y = dataTest['price_increase']
    X = dataTest[['Open', 'High', 'Low', 'Close', 'Adj Close', 'CloseUSD', 'Year', 'Month', 'Day']]


    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a random forest classifier
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_test_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Index NYA')
    #print('Test accuracy:', test_accuracy)
    NYAaccuracies.append(test_accuracy)
    # Make predictions for a new stock
    new_stock = pd.DataFrame([[100, 110, 90, 95, 100, 9500, 2030, 1+count, 3]], columns=X.columns)
    prediction = rf_model.predict(new_stock)
    listpred = prediction.tolist()
    NYApreds.append(listpred[0])
    print('Prediction:', prediction)
    count = count + 1
print('Most common prediction ', mode(NYApreds))
x = range(1, 11)

#graph accuracy
plt.plot(x, NYAaccuracies, 'o-')
plt.xlabel('Test Number')
plt.ylabel('Accuracy')
plt.title('Accuracy for NYA')
plt.show()  

#bar graph
plt.bar(x, NYApreds)
plt.xlabel('Test Number')
plt.ylabel('Prediction')
plt.title('NYA')
plt.show()
preds
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Testing
print(data.loc[[0]])
print(data.loc[[1]])
print(data.loc[[2]])
print(data.loc[[3]])
print(data.loc[[4]])
print(data.loc[[5]])
print(data.loc[[6]])
print(data.loc[[7]])
print(data.loc[[8]])
print(data.loc[[9]])
print(data.loc[[10]])
print(data.loc[[11]])
print(data.loc[[12]])

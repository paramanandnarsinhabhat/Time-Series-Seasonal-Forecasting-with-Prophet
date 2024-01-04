import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import mean 


import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv("/Users/paramanandbhat/Downloads/8._prophet/data/train_data.csv")
valid_data = pd.read_csv("/Users/paramanandbhat/Downloads/8._prophet/data/valid_data.csv")

print(train_data.shape)
print(train_data.head())

# Required Preprocessing 

train_data.timestamp = pd.to_datetime(train_data['Date'],format='%Y-%m-%d')
train_data.index = train_data.timestamp

valid_data.timestamp = pd.to_datetime(valid_data['Date'],format='%Y-%m-%d')
valid_data.index = valid_data.timestamp

plt.figure(figsize=(12,8))

plt.plot(train_data.index, train_data['count'], label='train_data')
plt.plot(valid_data.index,valid_data['count'], label='valid')
plt.legend(loc='best')
plt.title("Train and Validation Data")
plt.show()

from prophet import Prophet

print(train_data.head())

#Input in prophet needs to be date and target variable

df = train_data[['Date', 'count']]

df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])

df.head()

model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=184,freq="D")
forecast = model.predict(future)
fig = model.plot(forecast)

train_data.shape, valid_data.shape

print(len(forecast['yhat'][578:].values))

print(forecast)

valid_data['prophet'] =  forecast['yhat'][578:].values

plt.figure(figsize=(12,8))

plt.plot(train_data['count'],  label='train') 
plt.plot(valid_data['count'],  label='valid') 
plt.plot(valid_data['prophet'],  label='predicted') 
plt.legend(loc='best') 
plt.show()


# calculating RMSE 
rmse = sqrt(mean_squared_error(valid_data['count'], valid_data['prophet']))
print('The RMSE value for Prophet is', rmse)


forecast.index= forecast.ds

plt.figure(figsize=(12,8))

plt.plot(valid_data['count'],  label='valid') 
plt.plot(forecast['yhat_lower'][578:],  label='valid') 
plt.plot(forecast['yhat'][578:],  label='predicted') 
plt.plot(forecast['yhat_upper'][578:],  label='valid') 

plt.legend(loc='best') 
plt.show()
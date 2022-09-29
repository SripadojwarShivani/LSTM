#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

#reading the data 
#url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
dataset_train = pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv')

#prints all the rows and 1 col from training data
training_set = dataset_train.iloc[:, 1:2].values
training_set.shape

#printing first 5 values to get the clear view of the data.
#open - opening share for that day
#High - highest share in the day.
#low - lowest price of the day.
#last - last transaction went through the day.
#close - price shares ended that day.
dataset_train.head()

dataset_train.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled.shape

#converting the data into 3D array with the timestamps as 69 and X_train samples.
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    #converting x_train and y_train in array
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape,y_train.shape)
# (seq_len, batch_size, input_size), 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print("X_train:",X_train.shape)
print(X_train.shape[0])
print(X_train.shape[1])

model = Sequential()
#dimensionality of output space
#stacking LSTM layers so the consequent LSTM layer has a three-dimensional sequence input
#input_shape is the shape of the training dataset.

model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))

#dropout layer removes 20% data
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

#adam optimizer is used to compile the model
#loss is taken in mean_squared_error
model.compile(optimizer='adam',loss='mean_squared_error')

#epoch is number of times model works through training data
#batch_size=number of samples processed.
model.fit(X_train,y_train,epochs=100,batch_size=32)
model.summary()
#testing the data
dataset_test = pd.read_csv("https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_test.shape

#concatinating the train and test data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

#using 60 as time step again
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

#reshape such that it displays columns with index 1 and the last one.
inputs = inputs.reshape(-1,1)

#puts stock in readable format
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#plot for actual and predicting stock price
plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
#title of the plot
plt.title('TATA Stock Price Prediction')
#on horizontal x-axis
plt.xlabel('Time')
#on vertical y-axis
plt.ylabel('TATA Stock Price')
#display types of stock in plot
plt.legend()
plt.show()
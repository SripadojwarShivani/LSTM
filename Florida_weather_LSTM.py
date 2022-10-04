import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data 
florida = pd.read_csv("/home/shivanisri/Desktop/florida_file.csv")

#displays first 5 rows of the data
florida.head()

#displays the last 5 rows
florida.tail(5)

#converts string datatime to python datetime
florida["Date"] = pd.to_datetime(florida["Date"])

#bfill fills the nan values backwards
florida = florida[["Date", "Avg_Temp"]]
florida = florida.fillna(florida.bfill())
florida.columns = ['Date', 'Avg_Temp']
florida

#dividing data to train and test data
train = florida[:-225]
len(train)
test = florida[-225:]
len(test)
train_dates = pd.to_datetime(train['Date'])
test_dates  = pd.to_datetime(test['Date'])

# Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(train['Avg_Temp'].values.reshape(-1,1))

#examines 225 data and predicts the next one and same goes on
prediction_days = 225
len(scaled_data)
x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)
print(y_train.shape)

# Build The Model
model = Sequential()
model.add(LSTM(units =128, activation='relu', return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units =128, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units =128, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of the next value
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
history = model.fit(x_train, y_train, epochs = 25, batch_size=32, validation_split=0.1)

#plot
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.legend()
plt.show()

#Test the Model Accuracy on Existing Data
print(test.head())
actual_temp = test['Avg_Temp'].values
print("actual temp", actual_temp)
total_temp = pd.concat((train['Avg_Temp'], test['Avg_Temp']),axis=0)
print("total temp", total_temp)
model_inputs = total_temp[len(total_temp)-len(test)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)
print(len(model_inputs))

# Make Predictions on Test Data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)
print(pred)
from sklearn.metrics import mean_absolute_error
mean_absolute_error(test['Avg_Temp'], pred)
pred_ = pd.DataFrame(test['Date'])
pred_
pred_['Avg_Temp'] = pred
pred_["Date"] = pd.to_datetime(pred_["Date"])
print(pred_)
original = florida.loc[florida['Date'] >= '1990-01-01']
print(original)

#plot b/w actual and predicted temperature
import seaborn as sns
sns.lineplot(original['Date'], original['Avg_Temp'],label="actual")
sns.lineplot(pred_['Date'], pred_['Avg_Temp'],label="predicted")
plt.show()
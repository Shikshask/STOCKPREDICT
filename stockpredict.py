import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def features(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    return data

stock_data = pd.read_csv('AMZN.csv')
stock_data = features(stock_data)
features = stock_data[['Day', 'Month', 'Year']]
target = stock_data['Close']  
test_size = 0.2
split_index = int(len(stock_data) * (1 - test_size))
X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
y_train, y_test = target.iloc[:split_index], target.iloc[split_index:]

# Using a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

#Plotting the predicted outcome
plt.scatter(X_test.index, y_test, color='black', label='Actual Prices')
plt.scatter(X_test.index, predictions, color='blue', label='Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:07:42 2020
@description: This script analyzes stock data from Microsoft and creates a statistical analysis of it.

"""
# Libraries
import csv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Dataset & Preprocessing
full_y = [] # Open Stock Value (y-value)
full_x = np.reshape(range(0, 1259), (-1, 1))
f = open("MSFT.csv")

csv_f = csv.reader(f)

i = 0

for row in csv_f:
    if i != 0:
        full_y.append(float(row[1]))
    i += 1

train_x = np.reshape(range(0, 1000), (-1, 1))
test_x = np.reshape(range(1000, 1259), (-1, 1))
train_y = np.reshape(full_y[:1000], (-1, 1))
test_y = np.reshape(full_y[1000:], (-1, 1))
full_y = np.reshape(full_y, (-1, 1))

# Builidng the Model & Learning
model = LinearRegression()
model.fit(train_x, train_y)

# Predictions & Results
def graph(x, y):
    plt.plot(x, y, label = "Real Data")
    z = model.predict(x)        
    plt.plot(x, z, label = "Predictions")
    
    plt.xlabel('Days (Since Jan 1, 2013)') 
    plt.ylabel('Stock (MSFT) Open Price')
    plt.title('Statistical Analysis on MSFT Open Prices')
    plt.legend(loc="upper left")
    plt.show()

def mean_absolute_error(y, pred):
    a = 0
    b = 0
    for i in keras.losses.mean_absolute_error(y, pred):
        a += i
        b += 1
    return float(a)/b

graph(full_x, full_y)
print(mean_absolute_error(test_y, model.predict(test_x)))
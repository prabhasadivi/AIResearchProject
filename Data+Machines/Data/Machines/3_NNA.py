# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:25:33 2020
@description: This script analyzes stock data from Apple and creates a neural network model of it.

"""
# Libraries
import csv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

# Dataset & Preprocessing
full_y = [] # Open Stock Value (y-value)
full_x = np.reshape(range(0, 1259), (-1, 1))
f = open("AAPL.csv")

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

# Linear Model/Learning
model = keras.Sequential([
    keras.layers.Dense(64, activation = 'relu', input_shape=[1]),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_absolute_error',
    metrics=['accuracy']
    )

model.fit(train_x, train_y, epochs=10000)

train_loss, train_acc = model.evaluate(train_x, train_y)
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Loss on training data: {}%'.format(train_loss))
print('Loss on testing data: {}%'.format(test_loss))

def graph(x, y):
    plt.plot(x, y, label = "Real Data")
    z = model.predict(x)        
    plt.plot(x, z, label = "Predictions")
    
    plt.xlabel('Days (Since Jan 1, 2013)') 
    plt.ylabel('Stock (AAPL) Open Price')
    plt.title('Deep Neural Network Analysis on AAPL Open Prices')
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
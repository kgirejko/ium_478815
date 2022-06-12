import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import svm,  datasets
from sklearn.metrics import classification_report
scaler = StandardScaler()

# Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1) 

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

data = pd.read_csv('data.csv')
data.dropna()

training_data = data.sample(frac=0.9, random_state=25)
testing_data = data.drop(training_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")

training_data = training_data[['sqft_living', 'price']]

testing_data = testing_data[['sqft_living', 'price']]

training_data[['price']] = training_data[['price']] / 10000000
training_data[['sqft_living']] = training_data[['sqft_living']] / 10000

testing_data[['price']] = testing_data[['price']] / 10000000
testing_data[['sqft_living']] = testing_data[['sqft_living']] / 10000

# Tensory
X_training = training_data[['sqft_living']].to_numpy()
X_testing = testing_data[['sqft_living']].to_numpy()
y_training = training_data[['price']].to_numpy()
y_testing = testing_data[['price']].to_numpy()

import torch
torch.from_file
X_training = torch.from_numpy(X_training.astype(np.float32))
X_testing = torch.from_numpy(X_testing.astype(np.float32))
y_training = torch.from_numpy(y_training.astype(np.float32))
y_testing = torch.from_numpy(y_testing.astype(np.float32))

model = Model()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Trening
num_epochs = 1000
for epoch in range(num_epochs):
    y_predicted = model(X_training)
    loss = criterion(y_predicted,y_training)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch%100==0):
        print(f'epoch:{epoch+1},loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(X_testing)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_testing).sum()/float(y_testing.shape[0])
    #print(f'{acc:.4f}')

    rmse = mean_squared_error(y_testing, y_predicted)
#print(rmse)

mae = mean_absolute_error(y_testing, y_predicted)
#print(mae)

with open('metrics.txt', 'a+') as f:
    f.write('Root mean squared error:' + str(rmse) + '\n')  
    f.write('Mean absolute error:' + str(mae) + '\n')  
    #count = [float(line) for line in f if line]
    #builds = list(range(1, len(count)))
     
with open('metric.txt', 'a+') as f:
    f.write(str(rmse) + '\n')

with open('metric.txt') as file:
    y_rmse = [float(line) for line in file if line]
    x_builds = list(range(1, len(y_rmse) + 1))
            
plt.xlabel('Build')
plt.ylabel('RMSE')
plt.plot(x_builds, y_rmse, label='RMSE')
plt.legend()
plt.show()
plt.savefig('metrics.png')
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 04:13:37 2021

@author: shahi
"""

#importing the libraries

import numpy as np
import pandas as pd
#import matplotlib as plt
from matplotlib import pyplot as plt
#reading the dataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sklearn
import math
from sklearn.metrics import mean_absolute_error
# Importing the dataset
dataset = pd.read_csv('M.csv')
X = dataset.iloc[:, 13:-1].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#reshaping
# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Fast Charge KmH vs Electric Range(Training set)')
plt.xlabel('Fast charge KmH')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Fast Charge KmH vs Electric Range(Testing set)')
plt.xlabel('Fast charge KmH')
plt.ylabel('Electric Range')
plt.show()

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

numsum=0
densum=0
mean=0
summ=0
for i in range(0,21):
    summ=summ+Y_test[i]
    
mean=summ/21

for i in range(0,21):
    numsum=numsum+(Y_test[i]-y_pred[i])**2
    densum=densum+(Y_test[i]-mean)**2
    
R2=1-(numsum/densum)
#import math
Rmse = math.sqrt(numsum/21)
R2= r2_score(Y_test, y_pred)
mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
rmse = math.sqrt(mse)

# Importing the dataset
dataset = pd.read_csv('M.csv')
X = dataset.iloc[:, 12:-2].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Efficiency WhKm vs Electric Range(Training set)')
plt.xlabel('Efficiency WhKm')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Efficiency WhKm vs Electric Range(Testing set)')
plt.xlabel('Efficiency WhKm')
plt.ylabel('Electric Range')
plt.show()

# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

numsum=0
densum=0
mean=0
summ=0
for i in range(0,21):
    summ=summ+Y_test[i]
    
mean=summ/21

for i in range(0,21):
    numsum=numsum+(Y_test[i]-y_pred[i])**2
    densum=densum+(Y_test[i]-mean)**2
    
R2=1-(numsum/densum)
#import math
Rmse = math.sqrt(numsum/21)
R2= r2_score(Y_test, y_pred)
mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
rmse = math.sqrt(mse)

# Importing the dataset
dataset = pd.read_csv('M.csv')
X = dataset.iloc[:, 11:-3].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Battery pack KWH vs Electric Range(Training set)')
plt.xlabel('Battery pack KWH')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Battery pack KWH vs Electric Range(Testing set)')
plt.xlabel('Battery pack KWH')
plt.ylabel('Electric Range')
plt.show()

# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

numsum=0
densum=0
mean=0
summ=0
for i in range(0,21):
    summ=summ+Y_test[i]
    
mean=summ/21

for i in range(0,21):
    numsum=numsum+(Y_test[i]-y_pred[i])**2
    densum=densum+(Y_test[i]-mean)**2
    
R2=1-(numsum/densum)
#import math
Rmse = math.sqrt(numsum/21)
R2= r2_score(Y_test, y_pred)
mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
rmse = math.sqrt(mse)
# Importing the dataset
dataset = pd.read_csv('M.csv')
X = dataset.iloc[:, 10:-4].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Top_Speed KmH vs Electric Range(Training set)')
plt.xlabel('Top_Speed KmH')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Top_Speed KmH vs Electric Range(Testing set)')
plt.xlabel('Top_Speed KmH')
plt.ylabel('Electric Range')
plt.show()

# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

numsum=0
densum=0
mean=0
summ=0
for i in range(0,21):
    summ=summ+Y_test[i]
    
mean=summ/21

for i in range(0,21):
    numsum=numsum+(Y_test[i]-y_pred[i])**2
    densum=densum+(Y_test[i]-mean)**2
    
R2=1-(numsum/densum)
#import math
Rmse = math.sqrt(numsum/21)
R2= r2_score(Y_test, y_pred)
mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
rmse = math.sqrt(mse)

# Importing the dataset
dataset = pd.read_csv('M.csv')
X = dataset.iloc[:, 9:-5].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Accel Sec vs Electric Range(Training set)')
plt.xlabel('Accel Sec')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Accel Sec vs Electric Range(Testing set)')
plt.xlabel('Accel Sec')
plt.ylabel('Electric Range')
plt.show()

# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

numsum=0
densum=0
mean=0
summ=0
for i in range(0,21):
    summ=summ+Y_test[i]
    
mean=summ/21

for i in range(0,21):
    numsum=numsum+(Y_test[i]-y_pred[i])**2
    densum=densum+(Y_test[i]-mean)**2
    
R2=1-(numsum/densum)
#import math
Rmse = math.sqrt(numsum/21)
R2= r2_score(Y_test, y_pred)
mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
rmse = math.sqrt(mse)

# Importing the dataset
dataset = pd.read_csv('M.csv')
X = dataset.iloc[:, 8:-6].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Price Euro vs Electric Range(Training set)')
plt.xlabel('Price Euro')
plt.ylabel('Electric Range')
plt.show()


# Visualising the Polynomial Regression results (Testing)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title ('Price Euro vs Electric Range(Testing set)')
plt.xlabel('Price Euro')
plt.ylabel('Electric Range')
plt.show()

# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)


y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))
# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

numsum=0
densum=0
mean=0
summ=0
for i in range(0,21):
    summ=summ+Y_test[i]
    
mean=summ/21

for i in range(0,21):
    numsum=numsum+(Y_test[i]-y_pred[i])**2
    densum=densum+(Y_test[i]-mean)**2
    
R2=1-(numsum/densum)
#import math
Rmse = math.sqrt(numsum/21)
R2= r2_score(Y_test, y_pred)
mse = sklearn.metrics.mean_squared_error(Y_test, y_pred)
rmse = math.sqrt(mse)























































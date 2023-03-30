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

# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)

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


























































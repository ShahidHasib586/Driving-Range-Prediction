# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 12:53:37 2021

@author: shahi
"""

# Principal Component Analysis (PCA)

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('M.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# Taking care of missing data
# Updated Imputer
'''
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = 0, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 3:-1])
X[:, 3:-1]=missingvalues.transform(X[:, 3:-1])
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, :])
X[:, :]=missingvalues.transform(X[:, :])
'''

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le_1 = LabelEncoder()
X[:, 1] = le_1.fit_transform(X[:, 1])

le_0 = LabelEncoder()
X[:, 0] = le_0.fit_transform(X[:, 0])

le_2 = LabelEncoder()
X[:, 2] = le_2.fit_transform(X[:, 2])

le_3 = LabelEncoder()
X[:, 3] = le_3.fit_transform(X[:, 3])

le_4 = LabelEncoder()
X[:, 4] = le_4.fit_transform(X[:, 4])

le_5 = LabelEncoder()
X[:, 5] = le_5.fit_transform(X[:, 5])

le_6 = LabelEncoder()
X[:, 6] = le_1.fit_transform(X[:, 6])

# One Hot Encoding the "make" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct_0 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct_0.fit_transform(X))
ct_1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct_1.fit_transform(X))
ct_2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct_2.fit_transform(X))
ct_3 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct_3.fit_transform(X))
ct_4 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [4])], remainder='passthrough')
X = np.array(ct_4.fit_transform(X))
ct_5 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X = np.array(ct_5.fit_transform(X))
ct_6 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
X = np.array(ct_6.fit_transform(X))

#Avoiding dummy variable trap

X =X [:, 7:]
print(X)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#import statsmodels.formula.api as sm

import statsmodels.api as sm

X = np.append(arr = np.ones((102, 1)).astype(int), values = X, axis =1)
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,36,38,39,40,41,42,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,36,38,39,40,41,42,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,36,38,39,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,36,38,39,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,7,8,9,10,11,12,15,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,8,9,10,11,12,15,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,8,9,10,11,12,15,17,18,19,20,21,22,23,24,26,27,28,29,30,31,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,8,9,10,11,15,17,18,19,20,21,22,23,24,26,27,28,29,30,31,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,8,9,10,11,15,17,18,19,20,21,22,23,24,27,28,29,30,31,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5,6,8,9,10,11,15,17,18,19,20,21,22,23,24,27,29,30,31,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3,6,8,9,10,11,15,17,18,19,20,21,22,23,24,27,29,30,31,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3,6,8,9,10,11,15,17,18,19,20,21,23,24,27,29,30,31,33,34,36,38,40,41,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3,6,8,9,10,11,15,17,18,19,20,21,23,24,27,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3,6,8,9,10,15,17,18,19,20,21,23,24,27,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3,6,8,9,10,17,18,19,20,21,23,24,27,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2,6,8,9,10,17,18,19,20,21,23,24,27,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,6,8,9,10,17,18,19,20,21,23,24,27,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,6,8,9,17,18,19,20,21,23,24,27,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,6,8,9,18,19,20,21,23,24,27,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,6,8,9,18,19,20,23,24,27,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,6,8,9,18,19,20,23,24,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,6,8,9,18,19,23,24,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,6,8,9,19,23,24,29,30,31,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0, 1,6,8,9,19,23,24,29,30,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,8,9,19,23,24,29,30,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,8,9,19,23,24,30,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,8,19,23,24,30,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,19,24,30,33,34,36,38,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1,19,24,30,33,34,36,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,19,24,30,33,34,36,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,19,24,33,34,36,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,19,24,33,34,40,43,44,45]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Applying k-fold cross validation

from sklearn.model_selection import cross_val_score
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
accurracies = cross_val_score(estimator = regressor, X = X_train1, y = y_train, cv = 10 )

accurracies.mean()

accurracies.std()

import time
#errorin each value
for i in range(0,20):
    print("Error in value number", i, (y_test [i] - y_pred[i]))
time.sleep(1)

#combined rmse value
rss=((y_test-y_pred)**2).sum()
mse=np.mean((y_test-y_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((y_test-y_pred)**2)))

from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_test, y_pred)

accurracies.mean()















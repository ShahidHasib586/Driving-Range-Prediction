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

#feature scalling

#Applyring Simple linear regression 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

#Predicting the test set result
Y_pred = regressor.predict(X_test)

#visualizing the training set result

plt.scatter(X_train, Y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title ('Fast Charge vs Range(Training set)')
plt.xlabel('Fast charge')
plt.ylabel('range')
plt.show()

#visualizing the testing set result

plt.scatter(X_test, Y_test, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title ('Fast Charge vs Range(Testing set)')
plt.xlabel('Fast charge')
plt.ylabel('range')
plt.show()
# print the confidence intervals for the model coefficients
R2 = regressor.score(X, y)

print(R2)
#efficiency

# Importing the dataset
dataset1 = pd.read_csv('M.csv')
X1 = dataset1.iloc[:, 12:-2].values
y1 = dataset1.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

#feature scalling

#Applyring Simple linear regression 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X1_train, Y1_train)

#Predicting the test set result
Y1_pred = regressor.predict(X1_test)

#visualizing the training set result

plt.scatter(X1_train, Y1_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title ('Efficiency vs Range(Training set)')
plt.xlabel('Efficiency')
plt.ylabel('Range')
plt.show()

#visualizing the testing set result

plt.scatter(X1_test, Y1_test, color='red')

plt.plot(X1_train, regressor.predict(X1_train), color='blue')
plt.title ('Effeciency vs Range(Testing set)')
plt.xlabel('Effeciency charge')
plt.ylabel('Range')
plt.show()
# print the confidence intervals for the model coefficients
R2_1 = regressor.score(X1, y1)

print(R2_1)

# Battery Pack Kwh

X2 = dataset1.iloc[:, 11:-3].values
y2 = dataset1.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)

#feature scalling

#Applyring Simple linear regression 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X2_train, Y2_train)

#Predicting the test set result
Y2_pred = regressor.predict(X2_test)

#visualizing the training set result

plt.scatter(X2_train, Y2_train, color='red')

plt.plot(X2_train, regressor.predict(X2_train), color='blue')
plt.title ('Battery Pack Kwh vs Range(Training set)')
plt.xlabel('Battery Pack Kwh')
plt.ylabel('Range')
plt.show()

#visualizing the testing set result

plt.scatter(X2_test, Y2_test, color='red')

plt.plot(X2_train, regressor.predict(X2_train), color='blue')
plt.title ('Battery pack Kwh vs Range(Testing set)')
plt.xlabel('Battery Pack Kwh charge')
plt.ylabel('Range')
plt.show()
# print the confidence intervals for the model coefficients
R2_2 = regressor.score(X2, y2)

print(R2_2)

rss=((Y2_test-Y2_pred)**2).sum()
mse=np.mean((Y2_test-Y2_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((Y2_test-Y2_pred)**2)))
#Top Speed Kmh
X3 = dataset1.iloc[:, 10:-4].values
y3 = dataset1.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, y3, test_size = 0.2, random_state = 0)

#feature scalling

#Applyring Simple linear regression 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X3_train, Y3_train)

#Predicting the test set result
Y3_pred = regressor.predict(X3_test)

#visualizing the training set result

plt.scatter(X3_train, Y3_train, color='red')

plt.plot(X3_train, regressor.predict(X3_train), color='blue')
plt.title ('Top Speed Kmh vs Range(Training set)')
plt.xlabel('Top Speed Kmh charge')
plt.ylabel('range')
plt.show()

#visualizing the testing set result

plt.scatter(X3_test, Y3_test, color='red')

plt.plot(X3_train, regressor.predict(X3_train), color='blue')
plt.title ('Top Speed Kmh vs Range(Testing set)')
plt.xlabel('Top Speed Kmh charge')
plt.ylabel('range')
plt.show()
# print the confidence intervals for the model coefficients
R2_3 = regressor.score(X3, y3)

print(R2_3)

#AccelSec
X4 = dataset1.iloc[:, 9:-5].values
y4 = dataset1.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, y4, test_size = 0.2, random_state = 0)

#feature scalling

#Applyring Simple linear regression 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X4_train, Y4_train)

#Predicting the test set result
Y4_pred = regressor.predict(X4_test)

#visualizing the training set result

plt.scatter(X4_train, Y4_train, color='red')

plt.plot(X4_train, regressor.predict(X4_train), color='blue')
plt.title ('Accel Sec vs Range(Training set)')
plt.xlabel('Accel Sec charge')
plt.ylabel('range')
plt.show()

#visualizing the testing set result

plt.scatter(X4_test, Y4_test, color='red')

plt.plot(X4_train, regressor.predict(X4_train), color='blue')
plt.title ('Accel Sec vs Range(Testing set)')
plt.xlabel('Accel Sec charge')
plt.ylabel('range')
plt.show()
# print the confidence intervals for the model coefficients
R2_4 = regressor.score(X4, y4)

print(R2_4)

#PriceEuro
X5 = dataset1.iloc[:, 8:-6].values
y5 = dataset1.iloc[:, -1].values



from sklearn.model_selection import train_test_split

X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5, y5, test_size = 0.2, random_state = 0)

#feature scalling

#Applyring Simple linear regression 

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X5_train, Y5_train)

#Predicting the test set result
Y5_pred = regressor.predict(X5_test)

#visualizing the training set result

plt.scatter(X5_train, Y5_train, color='red')

plt.plot(X5_train, regressor.predict(X5_train), color='blue')
plt.title ('PriceEuro vs Range(Training set)')
plt.xlabel('PriceEuro charge')
plt.ylabel('range')
plt.show()

#visualizing the testing set result

plt.scatter(X5_test, Y5_test, color='red')

plt.plot(X5_train, regressor.predict(X5_train), color='blue')
plt.title ('PriceEuro vs Range(Testing set)')
plt.xlabel('PriceEuro charge')
plt.ylabel('range')
plt.show()
# print the confidence intervals for the model coefficients
R2_5 = regressor.score(X5, y5)

print(R2_5)




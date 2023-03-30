# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:51:18 2021

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


#Fast Charge vs Electric Range 
X = dataset.iloc[:, 13:-1].values
y = dataset.iloc[:, -1].values


#spliting the dataset into training and testing set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#reshaping

y = y.reshape(len(y),1)
print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
lin_reg_0 = SVR(kernel = 'rbf')
lin_reg_0.fit(X, y)

#Predicting the test set result
Y_pred = lin_reg_0.predict(X_test)

R0 = lin_reg_0.score(X, y)

print(R0)



# Visualising the Support Vector Regression results 
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(lin_reg_0.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title ('Fast Charge KmH vs Electric Range')
plt.xlabel('Fast charge KmH')
plt.ylabel('Electric Range')
plt.show()



#Efficiency WhKm vs Electric Range 
X1 = dataset.iloc[:, 12:-2].values
y1 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)

#reshaping

y1 = y1.reshape(len(y1),1)
print(y1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X1 = sc_X.fit_transform(X1)
y1 = sc_y.fit_transform(y1)
print(X1)
print(y1)



# Training the SVR model on the whole dataset
from sklearn.svm import SVR
lin_reg_1 = SVR(kernel = 'rbf')
lin_reg_1.fit(X1, y1)


#Predicting the test set result
Y1_pred = lin_reg_1.predict(X1_test)

R1 = lin_reg_1.score(X1, y1)

print(R1)



# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(sc_X.inverse_transform(X1)), max(sc_X.inverse_transform(X1)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X1), sc_y.inverse_transform(y1), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(lin_reg_1.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title ('Efficiency WhKm vs Electric Range')
plt.xlabel('Efficiency WhKm')
plt.ylabel('Electric Range')
plt.show()



#Battery pack KWH vs Electric Range 
X2 = dataset.iloc[:, 11:-3].values
y2 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)


#reshaping

y2 = y2.reshape(len(y2),1)
print(y2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X2 = sc_X.fit_transform(X2)
y2 = sc_y.fit_transform(y2)
print(X2)
print(y2)



# Training the SVR model on the whole dataset
from sklearn.svm import SVR
lin_reg_2 = SVR(kernel = 'rbf')
lin_reg_2.fit(X2, y2)


#Predicting the test set result
Y2_pred = lin_reg_2.predict(X2_test)

R2 = lin_reg_2.score(X2, y2)

print(R2)

rss=((Y2_test-Y2_pred)**2).sum()
mse=np.mean((Y2_test-Y2_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((Y2_test-Y2_pred)**2)))

# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(sc_X.inverse_transform(X1)), max(sc_X.inverse_transform(X1)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X2), sc_y.inverse_transform(y2), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(lin_reg_1.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title ('Battery pack KWH vs Electric Range')
plt.xlabel('Battery pack KWH')
plt.ylabel('Electric Range')
plt.show()



#Top_Speed KmH vs Electric Range 
X3 = dataset.iloc[:, 10:-4].values
y3 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, y3, test_size = 0.2, random_state = 0)


#reshaping

y3 = y3.reshape(len(y3),1)
print(y3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X3 = sc_X.fit_transform(X3)
y3 = sc_y.fit_transform(y3)
print(X3)
print(y3)



# Training the SVR model on the whole dataset
from sklearn.svm import SVR
lin_reg_3 = SVR(kernel = 'rbf')
lin_reg_3.fit(X3, y3)


#Predicting the test set result
Y3_pred = lin_reg_3.predict(X3_test)

R3 = lin_reg_3.score(X3, y3)

print(R3)



# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(sc_X.inverse_transform(X3)), max(sc_X.inverse_transform(X3)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X3), sc_y.inverse_transform(y3), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(lin_reg_1.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title ('Top_Speed KmH vs Electric Range')
plt.xlabel('Top_Speed KmH')
plt.ylabel('Electric Range')
plt.show()





#Accel Sec vs Electric Range 
X4 = dataset.iloc[:, 9:-5].values
y4 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, y4, test_size = 0.2, random_state = 0)


#reshaping

y4 = y4.reshape(len(y4),1)
print(y4)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X4 = sc_X.fit_transform(X4)
y4 = sc_y.fit_transform(y4)
print(X4)
print(y4)



# Training the SVR model on the whole dataset
from sklearn.svm import SVR
lin_reg_4 = SVR(kernel = 'rbf')
lin_reg_4.fit(X4, y4)


#Predicting the test set result
Y4_pred = lin_reg_4.predict(X4_test)

R4 = lin_reg_4.score(X4, y4)

print(R4)



# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(sc_X.inverse_transform(X4)), max(sc_X.inverse_transform(X4)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X4), sc_y.inverse_transform(y4), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(lin_reg_1.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title ('Accel Sec vs Electric Range')
plt.xlabel('Accel Sec')
plt.ylabel('Electric Range')
plt.show()



#Price Euro vs Electric Range 
X5 = dataset.iloc[:, 8:-6].values
y5 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5, y5, test_size = 0.2, random_state = 0)


#reshaping

y5 = y5.reshape(len(y5),1)
print(y5)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X5 = sc_X.fit_transform(X5)
y5 = sc_y.fit_transform(y5)
print(X5)
print(y5)



# Training the SVR model on the whole dataset
from sklearn.svm import SVR
lin_reg_5 = SVR(kernel = 'rbf')
lin_reg_5.fit(X5, y5)


#Predicting the test set result
Y5_pred = lin_reg_5.predict(X5_test)

R5 = lin_reg_5.score(X5, y5)

print(R5)



# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(sc_X.inverse_transform(X5)), max(sc_X.inverse_transform(X5)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X5), sc_y.inverse_transform(y5), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(lin_reg_1.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title ('Price Euro vs Electric Range')
plt.xlabel('Price Euro')
plt.ylabel('Electric Range')
plt.show()

print(R0)
print(R1)
print(R2)
print(R3)
print(R4)
print(R5)
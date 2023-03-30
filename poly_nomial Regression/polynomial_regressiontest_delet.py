

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



# Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.fit_transform(X_test)
lin_reg_0 = LinearRegression()
lin_reg_0.fit(X_poly_train, Y_train)

#Predicting the test set result
Y_pred = lin_reg_0.predict(X_poly_test)

R0 = lin_reg_0.score(X_poly, y)

print(R0)



# Visualising the Polynomial Regression results (Training)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_grid, lin_reg_0.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title ('Fast Charge KmH vs Electric Range(Training set)')
plt.xlabel('Fast charge KmH')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_grid, lin_reg_0.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title ('Fast Charge KmH vs Electric Range(Testing set)')
plt.xlabel('Fast charge KmH')
plt.ylabel('Electric Range')
plt.show()

#Efficiency WhKm vs Electric Range 
X1 = dataset.iloc[:, 12:-2].values
y1 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)


# Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X1_poly = poly_reg.fit_transform(X1)
X1_poly_train = poly_reg.fit_transform(X1_train)
X1_poly_test = poly_reg.fit_transform(X1_test)
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X1_poly_train, Y1_train)

#Predicting the test set result
Y1_pred = lin_reg_1.predict(X1_poly_test)

R1 = lin_reg_1.score(X1_poly, y1)

print(R1)



# Visualising the Polynomial Regression results (Training)
X1_grid = np.arange(min(X1), max(X1), 0.1)
X1_grid = X1_grid.reshape((len(X1_grid), 1))
plt.scatter(X1_train, Y1_train, color='red')
plt.plot(X1_grid, lin_reg_1.predict(poly_reg.fit_transform(X1_grid)), color = 'blue')
plt.title ('Efficiency WhKm vs Electric Range(Training set)')
plt.xlabel('Efficiency WhKm')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X1_grid = np.arange(min(X1), max(X1), 0.1)
X1_grid = X1_grid.reshape((len(X1_grid), 1))
plt.scatter(X1_test, Y1_test, color='red')
plt.plot(X1_grid, lin_reg_1.predict(poly_reg.fit_transform(X1_grid)), color = 'blue')
plt.title ('Efficiency WhKm vs Electric Range(Testing set)')
plt.xlabel('Efficiency WhKm')
plt.ylabel('Electric Range')
plt.show()

#Battery pack KWH vs Electric Range 
X2 = dataset.iloc[:, 11:-3].values
y2 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)


# Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 9)
X2_poly = poly_reg.fit_transform(X2)
X2_poly_train = poly_reg.fit_transform(X2_train)
X2_poly_test = poly_reg.fit_transform(X2_test)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X2_poly_train, Y2_train)

#Predicting the test set result
Y2_pred = lin_reg_2.predict(X2_poly_test)

R2 = lin_reg_2.score(X2_poly, y2)

print(R2)

rss=((Y2_test-Y2_pred)**2).sum()
mse=np.mean((Y2_test-Y2_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((Y2_test-Y2_pred)**2)))

# Visualising the Polynomial Regression results (Training)
X2_grid = np.arange(min(X2), max(X2), 0.1)
X2_grid = X2_grid.reshape((len(X2_grid), 1))
plt.scatter(X2_train, Y2_train, color='red')
plt.plot(X2_grid, lin_reg_2.predict(poly_reg.fit_transform(X2_grid)), color = 'blue')
plt.title ('Battery pack KWH vs Electric Range(Training set)')
plt.xlabel('Battery pack KWH')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X2_grid = np.arange(min(X2), max(X2), 0.1)
X2_grid = X2_grid.reshape((len(X2_grid), 1))
plt.scatter(X2_test, Y2_test, color='red')
plt.plot(X2_grid, lin_reg_2.predict(poly_reg.fit_transform(X2_grid)), color = 'blue')
plt.title ('Battery pack KWH vs Electric Range(Testing set)')
plt.xlabel('Battery pack KWH')
plt.ylabel('Electric Range')
plt.show()

#Top_Speed KmH vs Electric Range 
X3 = dataset.iloc[:, 10:-4].values
y3 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, y3, test_size = 0.2, random_state = 0)


# Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X3_poly = poly_reg.fit_transform(X3)
X3_poly_train = poly_reg.fit_transform(X3_train)
X3_poly_test = poly_reg.fit_transform(X3_test)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X3_poly_train, Y3_train)

#Predicting the test set result
Y3_pred = lin_reg_3.predict(X3_poly_test)

R3 = lin_reg_3.score(X3_poly, y3)

print(R3)



# Visualising the Polynomial Regression results (Training)
X3_grid = np.arange(min(X3), max(X3), 0.1)
X3_grid = X3_grid.reshape((len(X3_grid), 1))
plt.scatter(X3_train, Y3_train, color='red')
plt.plot(X3_grid, lin_reg_3.predict(poly_reg.fit_transform(X3_grid)), color = 'blue')
plt.title ('Top_Speed KmH vs Electric Range(Training set)')
plt.xlabel('Top_Speed KmH')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X3_grid = np.arange(min(X3), max(X3), 0.1)
X3_grid = X3_grid.reshape((len(X3_grid), 1))
plt.scatter(X3_test, Y3_test, color='red')
plt.plot(X3_grid, lin_reg_3.predict(poly_reg.fit_transform(X3_grid)), color = 'blue')
plt.title ('Top_Speed KmH vs Electric Range(Testing set)')
plt.xlabel('Top_Speed KmH')
plt.ylabel('Electric Range')
plt.show()



#Accel Sec vs Electric Range 
X4 = dataset.iloc[:, 9:-5].values
y4 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, y4, test_size = 0.2, random_state = 0)


# Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X4_poly = poly_reg.fit_transform(X4)
X4_poly_train = poly_reg.fit_transform(X4_train)
X4_poly_test = poly_reg.fit_transform(X4_test)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X4_poly_train, Y4_train)

#Predicting the test set result
Y4_pred = lin_reg_4.predict(X4_poly_test)

R4 = lin_reg_4.score(X4_poly, y4)

print(R4)



# Visualising the Polynomial Regression results (Training)
X4_grid = np.arange(min(X4), max(X4), 0.1)
X4_grid = X4_grid.reshape((len(X4_grid), 1))
plt.scatter(X4_train, Y4_train, color='red')
plt.plot(X4_grid, lin_reg_4.predict(poly_reg.fit_transform(X4_grid)), color = 'blue')
plt.title ('Accel Sec vs Electric Range(Training set)')
plt.xlabel('Accel Sec')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X4_grid = np.arange(min(X4), max(X4), 0.1)
X4_grid = X4_grid.reshape((len(X4_grid), 1))
plt.scatter(X4_test, Y4_test, color='red')
plt.plot(X4_grid, lin_reg_4.predict(poly_reg.fit_transform(X4_grid)), color = 'blue')
plt.title ('Accel Sec vs Electric Range(Testing set)')
plt.xlabel('Accel Sec')
plt.ylabel('Electric Range')
plt.show()

#Price Euro vs Electric Range 
X5 = dataset.iloc[:, 8:-6].values
y5 = dataset.iloc[:, -1].values


#spiliting the data into training and testig set
from sklearn.model_selection import train_test_split

X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5, y5, test_size = 0.2, random_state = 0)


# Training the Polynomial Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X5_poly = poly_reg.fit_transform(X5)
X5_poly_train = poly_reg.fit_transform(X5_train)
X5_poly_test = poly_reg.fit_transform(X5_test)
lin_reg_5 = LinearRegression()
lin_reg_5.fit(X5_poly_train, Y5_train)

#Predicting the test set result
Y5_pred = lin_reg_5.predict(X5_poly_test)

R5 = lin_reg_5.score(X5_poly, y5)

print(R5)



# Visualising the Polynomial Regression results (Training)
X5_grid = np.arange(min(X5), max(X5), 0.1)
X5_grid = X5_grid.reshape((len(X5_grid), 1))
plt.scatter(X5_train, Y5_train, color='red')
plt.plot(X5_grid, lin_reg_5.predict(poly_reg.fit_transform(X5_grid)), color = 'blue')
plt.title ('Price Euro vs Electric Range(Training set)')
plt.xlabel('Price Euro')
plt.ylabel('Electric Range')
plt.show()

# Visualising the Polynomial Regression results (Testing)
X5_grid = np.arange(min(X5), max(X5), 0.1)
X5_grid = X5_grid.reshape((len(X5_grid), 1))
plt.scatter(X5_test, Y5_test, color='red')
plt.plot(X5_grid, lin_reg_5.predict(poly_reg.fit_transform(X5_grid)), color = 'blue')
plt.title ('Price Euro vs Electric Range(Testing set)')
plt.xlabel('Price Euro')
plt.ylabel('Electric Range')
plt.show()



lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
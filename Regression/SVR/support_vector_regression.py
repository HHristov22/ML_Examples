"""
Support Vector Regression (SVR)

This code implements Support Vector Regression (SVR) to predict a continuous target variable.
SVR is a type of regression that uses the principles of Support Vector Machines (SVM), aiming to fit the best line within a margin of tolerance, allowing for some error.
The model is effective in handling non-linear relationships by using kernel functions (e.g., RBF kernel).
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./Regression/SVR/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

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
regressor = SVR(kernel = 'rbf') # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
regressor.fit(X, y)

#Linear regression
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)
y_pred = linear_regressor.predict(X)
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result for level N
N = 8
predicted_salary_N = sc_y.inverse_transform(regressor.predict(sc_X.transform([[N]])).reshape(-1,1))
print(f"The predicted salary for level {N} is: {round(predicted_salary_N[0, 0])}")
"""
Decision Tree Regression

This code implements Decision Tree Regression to predict a continuous target variable.
Decision Tree Regression works by splitting the data into subsets based on the feature values. 
At each split, it selects the feature that best separates the data, creating a tree structure of decision nodes and leaf nodes.
Each leaf node contains a prediction, which is the average value of the data points in that node.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('./Regression/Decision_Tree_Regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result for level N
N = 6.49
predicted_salary_N = regressor.predict([[N]])
print(f"The predicted salary for level {N} is: {round(predicted_salary_N[0], 2)}")

N = 6.50
predicted_salary_N = regressor.predict([[N]])
print(f"The predicted salary for level {N} is: {round(predicted_salary_N[0], 2)}")

N = 6.51
predicted_salary_N = regressor.predict([[N]])
print(f"The predicted salary for level {N} is: {round(predicted_salary_N[0], 2)}")
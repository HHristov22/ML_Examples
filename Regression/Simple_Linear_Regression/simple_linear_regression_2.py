# Simple Linear Regression

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
dataset = pd.read_csv('./Regression/Simple_Linear_Regression/Salary_Data.csv')

# Split dataset into features (YearsExperience) and target (Salary)
X = dataset[['YearsExperience']]
y = dataset['Salary']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Plot the dataset and the regression line
plt.scatter(X, y, color='blue', label='Actual dataset')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.title('Linear Regression: Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Function to predict salary based on years of experience
def predict_salary(N):
    return model.predict(np.array([[N]]))

# Example usage
input_years_experience = 6
predicted_salary = predict_salary(input_years_experience)
print(f"Yearly salary:{round(predicted_salary[0], 2)}")
print(f"Monthly salary:{round(predicted_salary[0]/12, 2)}")

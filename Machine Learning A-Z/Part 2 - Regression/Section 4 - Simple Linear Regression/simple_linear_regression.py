# Simple Linear Regression Model

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state = 0)

# Feature Scaling
# We do not need Feature Scaling in Linear Regression Model because Library we use in Linear Regression Model will itself take care of Feature Scaling.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predicting the Test Set Result
Y_pred = regressor.predict(X_test)


# Visualising the Training Set Results
plt.figure(1)                   # So, that we can see both Training Set and Test Set Graphs at same time
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green' )
plt.title('Experience vs Salary (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


# Visualising the Twst Set Results
plt.figure(2)                   # So, that we can see both Training Set and Test Set Graphs at same time
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green' )             # This line may remain the same so that we can compare the model which we trained on training data set with the new test values
plt.title('Experience vs Salary (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')


plt.show()                      # If we want to show both the plot at the same time, so that we can compare we have to show() it only once.
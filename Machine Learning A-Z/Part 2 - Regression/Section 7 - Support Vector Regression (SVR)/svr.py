# Support Vector Regression
# In SVR, we are supposed to do Feature Scaling on our own. Because, the library which we use for SVR does not suppports automatic feature scaling, that's why we have to do feature scaling in SVR.


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("D:\GitHub_Repositories\Learn-Machine-Learning\Machine Learning A-Z\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = X.reshape(-1,1)             # Added by me
Y = Y.reshape(-1,1)             # Added by me
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Fitting the SVR to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

# Predicting a new result
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))               # Brought the prediction back to the original scale using "inverse_transform".  Converted 6.5 to Array, because "transform" takes parameter as an array.


# Visualising the SVR results                                           # Salary of CEO in the dataset is too much as compared to others. Therefore, CEO is set as an outlier by the SVR and calculated according to other position points.
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
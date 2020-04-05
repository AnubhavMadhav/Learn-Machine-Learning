# Multiple Linear Regression Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("D:\GitHub_Repositories\Learn-Machine-Learning\Machine Learning A-Z\Part 2 - Regression\Section 5 - Multiple Linear Regression\Multiple_Linear_Regression\Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])         # Cause 'State'(Categorical Variable) is at index '3'          # Encoding them with labels like 0,1,2
ct = ColumnTransformer([('onehotencoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')         # Creating Dummy Variables.
X = ct.fit_transform(X)
# Here, we do not need to encode the Y(dependent variable) because it is not categorized (it has linear values)


# Avoiding the Dummy Variable Trap
X = X[:, 1:]                # Though, the library we will use will itself take care of the Dummy Variable Trap, but still we can do this


# Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

# Feature Scaling
# We do not need Feature Scaling in Linear Regression Model because Library we use in Linear Regression Model will itself take care of Feature Scaling.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Fitting the Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predicting the Test Set Results
Y_pred = regressor.predict(X_test)                  # Creating a Vector


# Building the optimal model using Backward Elimination
# Till above, we used all the columns/variables in our model, but there might be some variables/columns which are of no use predict are dependent variable, i.e. some columns/variable do not affects deppendent variable/column much.
# To, ignore and not use those unnecessary variables/columns in our model, we will use Backward Elimination (we can use other methods too like forward selection or bidirectional elimination but Backward Elimination is faster than rest)
# We want only those Independent variables/columns in 'X' those who have High Impact on Profit(Dependent Variable)
# Multiple Linear Regression Model: y = b0 + b1*x1 + b2*x2 + ...... + bn*xn
# The library which we are going to use do not counts the constant(b0) because it is not associated with any column/variable, but we need that constant otherwise results will ne wrong.
# To make our library count that constant(b0), we will associate it with x0, where x0=1.
# We need to make it something like: y = b0*x0 + b1*x1 + b2*x2 + ...... + bn*xn, where x0=1
# Therefore, we will add a new column/variable 'x0' to our 'X'(Independent Variable) using numpy.


# Backward Elimination: Starting with all the Independent Variables in 'X' , we will keep on removing all those independent variables who are statistically insignificant

import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# 'axis = 0' - adds a row
# 'axis = 1' - adds a column
# Here, we are adding 'X' to Array of ones() because we need ones() at the beginning of 'X'. Therefore, we are not adding ones() to 'X'.


# Very Very Important and Updated Part
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(int)               # Converted fron object type to int type, otherwise OLS does not works on object type
# Ordinary Least Squares
regressor_OLS = sm.OLS(Y, X_opt).fit()
# Significance Level = 5% i.e. 0.05
regressor_OLS.summary()             # Reading from the summary, observe the highest p-value among all columns
# consider the predictor with the highest p-value, if p-value>SL, remove that predictor/variable else go to FInish(Our Model is Ready)


X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(int)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
X_opt = X_opt.astype(int)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
X_opt = X_opt.astype(int)
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()


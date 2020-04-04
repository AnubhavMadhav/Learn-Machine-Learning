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
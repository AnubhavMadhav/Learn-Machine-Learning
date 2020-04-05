# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("D:\GitHub_Repositories\Learn-Machine-Learning\Machine Learning A-Z\Part 2 - Regression\Section 6 - Polynomial Regression\Position_Salaries.csv")
# Here, we need 'X" as a Matrix and 'Y' as a Vector, so that there may be no problem as we move further
X = dataset.iloc[:, 1:2].values                 # Using "1:2" makes Matrix whereas if we would have used "1", it makes a Vector
Y = dataset.iloc[:, 2].values                   # Using "2" makes Vector wheras if we wpuld have used "2:3", it would have made a Matrix

# There is no missing data in the dataset
# # Taking Care of Missing Data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')         # default missing_val ="np.NaN" and default strategy = 'mean' and default axis = 0
'''
missing_values is "np.NaN" by default or else can be an Integer
strategy can be: 1) 'mean'          2) 'median'             3) 'most_frequent'
This is for older version: axis can be :    1) 0 - 'strategy will be performed along column'               2) 1 - 'strategy will be performed along row'
'''
# imputer = imputer.fit(X[:,1:3])                 # The "1:3" on the right of ',' represents "1 and 2" column index
# X[:,1:3] = imputer.transform(X[:,1:3])          # Transforming it in the form of table or matrix or dataset


# Our dataset is well and no need to be encoded
# Encoding Categorical Data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# labelencoder_X = LabelEncoder()
# X[:,0] = labelencoder_X.fit_transform(X[:,0])
# # 3 countries are given 0,1,2. Machine may compare them mathematically for large, medium and small. To ignore that, we need Dummy Variable.
#
# ct = ColumnTransformer([('onehotencoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')         # Creating Dummy Variables.
# # 3 different columns for 3 different categories are created and marked as '1' in the row they are present and '0' in the other rows
# X = ct.fit_transform(X)
# # onehotencoder = OneHotEncoder(categories=[0])             # Old Method
# # X = onehotencoder.fit_transform(X).toarray()
# labelencoder_Y = LabelEncoder()
# Y = labelencoder_Y.fit_transform(Y)

# Our dataset is too small and have only 10 rows, and we need a good accuracy so it will be better if we use the whole dataset for training and do not split it into test dataset
# Splitting the Dataset into Training Set and Test Set
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
# if we have entered the 'test_size' then there is no need to enter 'train_size', because "train_size = 1 - test_size" always

# Our dataset is already good and library which we are using is "Linear" as that in simple and multiple, therefore, no need of feature scaling
# Feature Scaling
# It is used to bring all the variables in same scale i.e. they should not vary largely, cause if they vary largely, then it may end up giving error in models which include Euclidian's Distance and even if we do not use Euclidian's Distance in our model, it will be better if we feature scale our data so that it will work faster.
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()                         # Bringing all to the scale of "-1 to 1"
# X_train = sc_X.fit_transform(X_train)               # We always need to fit and transform the training dataset
# X_test = sc_X.transform(X_test)                     # We just need to transform the test data set and no need to fit it, cause X_train is already fitted
# # We do not need to scale the 'Y'(dependent) variable cause it is already 'categorical' i.e. it is a classification problem whereas in some cases like regression we may need to feature scale the dependent variable too
# sc_Y = StandardScaler()
# Y_train = sc_Y.fit_transform(Y_train)



# Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)


# Fitting Polynomial Regression to the Dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)                   # Keep on changing the degrees to see that which degree suits best to model by viewing it on graph for each degree. Here, answer is '5'            # Default: degree = 2
X_poly = poly_reg.fit_transform(X)                          # This will create a column of ones() as well at index 0 as a coefficient for 'b0', as we did in multiple linear regression using numpy.
# transformed our original matrix of features 'X' into our new matrix of features 'X_poly' containing the original independent variables position levels and it's associated polynomial terms

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)


# Visualizing the Linear Regression Model
plt.figure(1)
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')



# Visualizing the Polynomial Regression Model
plt.figure(2)
X_grid = np.arange(min(X), max(X), 0.1)                         # This is a Vector.  # To see the continuous curve
X_grid = X_grid.reshape((len(X_grid), 1))                       # Converting Vector to Matrix
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'green')                   # Don't use 'X_poly' directly instead use "poly_reg.fit_transform(X)"
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')




plt.show()





# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])



# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


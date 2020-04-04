# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("D:\GitHub_Repositories\Learn-Machine-Learning\Machine Learning A-Z\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Data.csv")

X = dataset.iloc[:,:-1].values              # For Independent Columns              # ':' before ',' represents 'all-rows' and ':-1' right to ',' represents 'all columns except last column'
Y = dataset.iloc[:, 3].values               # For Dependent Column (Last Col in Data.csv)                   ':' before ',' represents 'all-rows' and  '3' right to ',' represents the index of the last column


# Taking Care of Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')         # default missing_val ="np.NaN" and default strategy = 'mean' and default axis = 0
'''
missing_values is "np.NaN" by default or else can be an Integer
strategy can be: 1) 'mean'          2) 'median'             3) 'most_frequent'
This is for older version: axis can be :    1) 0 - 'strategy will be performed along column'               2) 1 - 'strategy will be performed along row'
'''
imputer = imputer.fit(X[:,1:3])                 # The "1:3" on the right of ',' represents "1 and 2" column index
X[:,1:3] = imputer.transform(X[:,1:3])          # Transforming it in the form of table or matrix or dataset



# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# 3 countries are given 0,1,2. Machine may compare them mathematically for large, medium and small. To ignore that, we need Dummy Variable.

ct = ColumnTransformer([('onehotencoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')         # Creating Dummy Variables.
# 3 different columns for 3 different categories are created and marked as '1' in the row they are present and '0' in the other rows
X = ct.fit_transform(X)
# onehotencoder = OneHotEncoder(categories=[0])             # Old Method
# X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
# if we have entered the 'test_size' then there is no need to enter 'train_size', because "train_size = 1 - test_size" always

# Feature Scaling
# It is used to bring all the variables in same scale i.e. they should not vary largely, cause if they vary largely, then it may end up giving error in models which include Euclidian's Distance and even if we do not use Euclidian's Distance in our model, it will be better if we feature scale our data so that it will work faster.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()                         # Bringing all to the scale of "-1 to 1"
X_train = sc_X.fit_transform(X_train)               # We always need to fit and transform the training dataset
X_test = sc_X.transform(X_test)                     # We just need to transform the test data set and no need to fit it, cause X_train is already fitted
# We do not need to scale the 'Y'(dependent) variable cause it is already 'categorical' i.e. it is a classification problem whereas in some cases like regression we may need to feature scale the dependent variable too

print(X)
print(Y)

'''
# Necessary: 
Importing the libraries
Importing the dataset
Splitting the Dataset into Training Set and Test Set
Feature Scaling - (not everytime but only in some models)


# Not Necessary:
Taking Care of Missing Data
Encoding Categorical Data
'''
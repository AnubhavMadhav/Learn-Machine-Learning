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
imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')         # default missing_val ="NaN" and default strategy = 'mean' and default axis = 0
'''
missing_values is "np.NaN" by default or else can be an Integer
strategy can be: 1) 'mean'          2) 'median'             3) 'most_frequent'
This is for older version: axis can be :    1) 0 - 'strategy will be performed along column'               2) 1 - 'strategy will be performed along row'
'''
imputer = imputer.fit(X[:,1:3])                 # The "1:3" on the right of ',' represents "1 and 2" column index
X[:,1:3] = imputer.transform(X[:,1:3])          # Transforming it in the form of table or matrix or dataset

print(X)
print(Y)
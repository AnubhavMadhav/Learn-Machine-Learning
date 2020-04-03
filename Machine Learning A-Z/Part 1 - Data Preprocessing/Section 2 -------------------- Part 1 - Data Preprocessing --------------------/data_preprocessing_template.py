# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("D:\GitHub_Repositories\Learn-Machine-Learning\Machine Learning A-Z\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Data.csv")

X = dataset.iloc[:,:-1].values              # For Independent Columns              # ':' before ',' represents 'all-rows' and ':-1' right to ',' represents 'all columns except last column'
Y = dataset.iloc[:, 3].values               # For Dependent Column (Last Col in Data.csv)                   ':' before ',' represents 'all-rows' and  '3' right to ',' represents the index of the last column

print(X)
print(Y)
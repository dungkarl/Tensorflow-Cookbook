"""
    + Working with Nearest Neighbors
    + Working with Text-Based Distances
    + Computing Mixed Distance Functions
    + Using an Address Matching Example
    + Using Nearest Neighbors for Image Recognition
"""
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import requests

tf.reset_default_graph()
sess = tf.Session()


# prepare data

housing_url = 'https://archive.ics.uci.edu/ml/machine-learningdatabases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)
# Request data
housing_file = requests.get(housing_url)
# Parse Data
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]

# 3:
y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

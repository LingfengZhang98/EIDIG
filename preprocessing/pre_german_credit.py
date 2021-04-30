"""
This python file preprocesses the German Credit Dataset.
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

"""
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Q8MAW8
"""

# make outputs stable across runs
np.random.seed(42)
tf.random.set_seed(42)


# load german credit risk dataset
data_path = ('datasets/proc_german_num_02 withheader-2.csv')
df = pd.read_csv(data_path)


# preprocess data
data = df.values.astype(np.int32)
data[:,0] = (data[:,0]==1).astype(np.int64)
bins_loan_nurnmonth = [0] + [np.percentile(data[:,2], percent, axis=0) for percent in [25, 50, 75]] + [80]
bins_creditamt = [0] + [np.percentile(data[:,4], percent, axis=0) for percent in [25, 50, 75]] + [200]
bins_age = [15, 25, 45, 65, 120]
list_index_num = [2, 4, 10]
list_bins = [bins_loan_nurnmonth, bins_creditamt, bins_age]
for index, bins in zip(list_index_num, list_bins):
    data[:, index] = np.digitize(data[:, index], bins, right=True)


# split data into training data and test data
X = data[:, 1:]
y = data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# set constraints for each attribute, 839808 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T


# for german credit data, gender(6) and age(9) are protected attributes in 24 features
protected_attribs = [6, 9]
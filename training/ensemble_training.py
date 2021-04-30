"""
This python file trains the ensemble models for relabeling.
Five classifiers are trained for majority voting, including KNN, MLP, RBF SVM, Random Forest, Naive Bayes.
"""


import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census_income, pre_german_credit, pre_bank_marketing


# collect datasets
datasets = [(pre_census_income.X, pre_census_income.y), (pre_german_credit.X, pre_german_credit.y), (pre_bank_marketing.X, pre_bank_marketing.y)]
names = ['adult', 'german', 'bank']


# create classifiers
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
nb_clf = GaussianNB()


# ensemble above classifiers for majority voting
eclf = VotingClassifier(estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svm', svm_clf), ('rf', rf_clf), ('nb', nb_clf)],
                        voting='soft')


# set a pipeline to handle the prediction process
clf = Pipeline([('scaler', StandardScaler()),
                ('ensemble', eclf)])


# train, evaluate and save ensemble models for each dataset
for i, ds in enumerate(datasets):
    model = clone(clf)
    X, y = ds
    if i == 0:
        X = np.delete(X, pre_census_income.protected_attribs, axis=1)
    elif i == 1:
        X = np.delete(X, pre_german_credit.protected_attribs, axis=1)
    else:
        X = np.delete(X, pre_bank_marketing.protected_attribs, axis=1)
    if len(X) > 10000:
        split_ratio = 0.2
    else:
        split_ratio = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(names[i] + ':', score)
    joblib.dump(model, 'models/models_from_tests/' + names[i] + '_ensemble.pkl')


"""
accuraccy w.r.t. test data:
adult: 84.07%
german: 76.5%
bank: 89.25%
"""
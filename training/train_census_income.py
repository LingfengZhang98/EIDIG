"""
This python file constructs and trains the model for Census Income Dataset.
"""


import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census_income
import tensorflow as tf
from tensorflow import keras


# create and train a six-layer neural network for the binary classification task
model = keras.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=pre_census_income.X_train.shape[1:]),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(15, activation="relu"),
    keras.layers.Dense(15, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])

# uncomment for training
"""
history = model.fit(pre_census_income.X_train, pre_census_income.y_train, epochs=30, validation_data=(pre_census_income.X_val, pre_census_income.y_val))
model.evaluate(pre_census_income.X_test, pre_census_income.y_test) # 84.32% accuracy
model.save("models/models_from_tests/adult_model.h5")
"""

# The precision rate is  0.7338425381903643 , the recall rate is  0.5454148471615721 , and the F1 score is  0.625751503006012
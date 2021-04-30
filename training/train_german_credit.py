"""
This python file constructs and trains the model for German Credit Dataset.
"""


import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_german_credit
import tensorflow as tf
from tensorflow import keras


# create and train a six-layer neural network for the binary classification task
model = keras.Sequential([
    keras.layers.Dense(50, activation="relu", input_shape=pre_german_credit.X_train.shape[1:]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(15, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])

# uncomment for training
"""
history = model.fit(pre_german_credit.X_train, pre_german_credit.y_train, epochs=20)
model.evaluate(pre_german_credit.X_test, pre_german_credit.y_test) # 78.25% accuracy
model.save("models/models_from_tests/german_model.h5")
"""

# The precision rate is  0.6781609195402298 , the recall rate is  0.5 , and the F1 score is  0.5756097560975609
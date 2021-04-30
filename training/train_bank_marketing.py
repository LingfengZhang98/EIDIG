"""
This python file constructs and trains the model for Bank Marketing Dataset.
"""


import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_bank_marketing
import tensorflow as tf
from tensorflow import keras


# create and train a six-layer neural network for the binary classification task
model = keras.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=pre_bank_marketing.X_train.shape[1:]),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(15, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(5, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])


# uncomment for training
"""
history = model.fit(pre_bank_marketing.X_train, pre_bank_marketing.y_train, epochs=30, validation_data=(pre_bank_marketing.X_val, pre_bank_marketing.y_val))
model.evaluate(pre_bank_marketing.X_test, pre_bank_marketing.y_test) # 89.22% accuracy
model.save("models/models_from_tests/bank_model.h5")
"""

# The precision rate is  0.7181467181467182 , the recall rate is  0.17048579285059579 , and the F1 score is  0.27555555555555555
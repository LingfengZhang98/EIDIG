"""
This python file retrains the original models with augmented training set.
"""


import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import joblib
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
import train_census_income
import train_german_credit
import train_bank_marketing

def retraining(dataset_name, approach_name, ids):
    # randomly sample 5% of individual discriminatory instances generated for data augmentation
    # then retrain the original models

    ensemble_clf = joblib.load('models/ensemble_models/' + dataset_name + '_ensemble.pkl')
    if dataset_name == 'adult':
        protected_attribs = pre_census_income.protected_attribs
        X_train = pre_census_income.X_train_all
        y_train = pre_census_income.y_train_all
        X_test = pre_census_income.X_test
        y_test = pre_census_income.y_test
        model = train_census_income.model
    elif dataset_name == 'german':
        protected_attribs = pre_german_credit.protected_attribs
        X_train = pre_german_credit.X_train
        y_train = pre_german_credit.y_train
        X_test = pre_german_credit.X_test
        y_test = pre_german_credit.y_test
        model = train_german_credit.model
    elif dataset_name == 'bank':
        protected_attribs = pre_bank_marketing.protected_attribs
        X_train = pre_bank_marketing.X_train_all
        y_train = pre_bank_marketing.y_train_all
        X_test = pre_bank_marketing.X_test
        y_test = pre_bank_marketing.y_test
        model = train_bank_marketing.model
    ids_aug = np.empty(shape=(0, len(X_train[0])))
    num_aug = int(len(ids) * 0.05)
    for _ in range(num_aug):
        rand_index = np.random.randint(len(ids))
        ids_aug = np.append(ids_aug, [ids[rand_index]], axis=0)
    label_vote = ensemble_clf.predict(np.delete(ids_aug, protected_attribs, axis=1))
    X_train = np.append(X_train, ids_aug, axis=0)
    y_train = np.append(y_train, label_vote, axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    model.evaluate(X_test, y_test)
    model.save('models/models_from_tests/' + dataset_name + '_' + approach_name + '_retrained_model.h5')


# census income
ids_C_a_ADF = np.load('logging_data/generated discriminatory instances/C-a_ids_ADF.npy')
ids_C_r_ADF = np.load('logging_data/generated discriminatory instances/C-r_ids_ADF.npy')
ids_C_g_ADF = np.load('logging_data/generated discriminatory instances/C-g_ids_ADF.npy')
ids_C_a_r_ADF = np.load('logging_data/generated discriminatory instances/C-a&r_ids_ADF.npy')
ids_C_a_g_ADF = np.load('logging_data/generated discriminatory instances/C-a&g_ids_ADF.npy')
ids_C_r_g_ADF = np.load('logging_data/generated discriminatory instances/C-r&g_ids_ADF.npy')
C_ids_ADF = np.concatenate((ids_C_a_ADF, ids_C_r_ADF, ids_C_g_ADF, ids_C_a_r_ADF, ids_C_a_g_ADF, ids_C_r_g_ADF), axis=0)
ids_C_a_EIDIG_5 = np.load('logging_data/generated discriminatory instances/C-a_ids_EIDIG_5.npy')
ids_C_r_EIDIG_5 = np.load('logging_data/generated discriminatory instances/C-r_ids_EIDIG_5.npy')
ids_C_g_EIDIG_5 = np.load('logging_data/generated discriminatory instances/C-g_ids_EIDIG_5.npy')
ids_C_a_r_EIDIG_5 = np.load('logging_data/generated discriminatory instances/C-a&r_ids_EIDIG_5.npy')
ids_C_a_g_EIDIG_5 = np.load('logging_data/generated discriminatory instances/C-a&g_ids_EIDIG_5.npy')
ids_C_r_g_EIDIG_5 = np.load('logging_data/generated discriminatory instances/C-r&g_ids_EIDIG_5.npy')
C_ids_EIDIG_5 = np.concatenate((ids_C_a_EIDIG_5, ids_C_r_EIDIG_5, ids_C_g_EIDIG_5, ids_C_a_r_EIDIG_5, ids_C_a_g_EIDIG_5, ids_C_r_g_EIDIG_5), axis=0)
ids_C_a_EIDIG_INF = np.load('logging_data/generated discriminatory instances/C-a_ids_EIDIG_INF.npy')
ids_C_r_EIDIG_INF = np.load('logging_data/generated discriminatory instances/C-r_ids_EIDIG_INF.npy')
ids_C_g_EIDIG_INF = np.load('logging_data/generated discriminatory instances/C-g_ids_EIDIG_INF.npy')
ids_C_a_r_EIDIG_INF = np.load('logging_data/generated discriminatory instances/C-a&r_ids_EIDIG_INF.npy')
ids_C_a_g_EIDIG_INF = np.load('logging_data/generated discriminatory instances/C-a&g_ids_EIDIG_INF.npy')
ids_C_r_g_EIDIG_INF = np.load('logging_data/generated discriminatory instances/C-r&g_ids_EIDIG_INF.npy')
C_ids_EIDIG_INF = np.concatenate((ids_C_a_EIDIG_INF, ids_C_r_EIDIG_INF, ids_C_g_EIDIG_INF, ids_C_a_r_EIDIG_INF, ids_C_a_g_EIDIG_INF, ids_C_r_g_EIDIG_INF), axis=0)


# german credit
ids_G_g_ADF = np.load('logging_data/generated discriminatory instances/G-g_ids_ADF.npy')
ids_G_a_ADF = np.load('logging_data/generated discriminatory instances/G-a_ids_ADF.npy')
ids_G_g_a_ADF = np.load('logging_data/generated discriminatory instances/G-g&a_ids_ADF.npy')
G_ids_ADF = np.concatenate((ids_G_g_ADF, ids_G_a_ADF, ids_G_g_a_ADF), axis=0)
ids_G_g_EIDIG_5 = np.load('logging_data/generated discriminatory instances/G-g_ids_EIDIG_5.npy')
ids_G_a_EIDIG_5 = np.load('logging_data/generated discriminatory instances/G-a_ids_EIDIG_5.npy')
ids_G_g_a_EIDIG_5 = np.load('logging_data/generated discriminatory instances/G-g&a_ids_EIDIG_5.npy')
G_ids_EIDIG_5 = np.concatenate((ids_G_g_EIDIG_5, ids_G_a_EIDIG_5, ids_G_g_a_EIDIG_5), axis=0)
ids_G_g_EIDIG_INF = np.load('logging_data/generated discriminatory instances/G-g_ids_EIDIG_INF.npy')
ids_G_a_EIDIG_INF = np.load('logging_data/generated discriminatory instances/G-a_ids_EIDIG_INF.npy')
ids_G_g_a_EIDIG_INF = np.load('logging_data/generated discriminatory instances/G-g&a_ids_EIDIG_INF.npy')
G_ids_EIDIG_INF = np.concatenate((ids_G_g_EIDIG_INF, ids_G_a_EIDIG_INF, ids_G_g_a_EIDIG_INF), axis=0)


# bank marketing
B_ids_ADF = np.load('logging_data/generated discriminatory instances/B-a_ids_ADF.npy')
B_ids_EIDIG_5 = np.load('logging_data/generated discriminatory instances/B-a_ids_EIDIG_5.npy')
B_ids_EIDIG_INF = np.load('logging_data/generated discriminatory instances/B-a_ids_EIDIG_INF.npy')


# retrain the original models
retraining('adult', 'ADF', C_ids_ADF) # The accuracy is 83.66%, the precision rate is  0.7543988269794721 , the recall rate is  0.4493449781659389 , and the F1 score is  0.5632183908045977
retraining('german', 'ADF', G_ids_ADF) # The accuracy is 77.25%, the precision rate is  0.6588235294117647 , the recall rate is  0.4745762711864407 , and the F1 score is  0.5517241379310346
retraining('bank', 'ADF', B_ids_ADF) # The accuracy is 89.03%, the precision rate is  0.6234413965087282 , the recall rate is  0.229147571035747 , and the F1 score is  0.3351206434316354

retraining('adult', 'EIDIG_5', C_ids_EIDIG_5) # The accuracy is 84.12%, the precision rate is  0.7149505526468877 , the recall rate is  0.5366812227074236 , and the F1 score is  0.6131204789224245
retraining('german', 'EIDIG_5', G_ids_EIDIG_5) # The accuracy is 77.00%, the precision rate is  0.6413043478260869 , the recall rate is  0.5 , and the F1 score is  0.5619047619047619
retraining('bank', 'EIDIG_5', B_ids_EIDIG_5) # The accuracy is 89.14%, the precision rate is  0.6543909348441926 , the recall rate is  0.21173235563703025 , and the F1 score is  0.31994459833795014

retraining('adult', 'EIDIG_INF', C_ids_EIDIG_INF) # The accuracy is 84.11%, the precision rate is  0.6909937888198758 , the recall rate is  0.5829694323144105 , and the F1 score is  0.6324017053529134
retraining('german', 'EIDIG_INF', G_ids_EIDIG_INF) # The accuracy is 74.50%, the precision rate is  0.5655737704918032 , the recall rate is  0.5847457627118644 , and the F1 score is  0.575
retraining('bank', 'EIDIG_INF', B_ids_EIDIG_INF) # The accuracy is 89.00%, the precision rate is  0.6224489795918368 , the recall rate is  0.2236480293308891 , and the F1 score is  0.32906271072151044
"""
This python file evaluates the classification performance of models, such as precision and recall.
"""


import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
from tensorflow import keras
import sys
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing


def precision_recall(y_true, y_pred):
    # evaluate precision, recall, and F1 score

    true_positive = np.dot(y_true, y_pred)
    false_positive = np.sum((y_true - y_pred) == -1.0)
    false_negative = np.sum((y_true - y_pred) == 1.0)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1score = 2 * precision * recall / (precision + recall)
    print('The precision rate is ', precision, ', the recall rate is ', recall, ', and the F1 score is ', F1score)


for dataset_name, dataset_module in [('adult', pre_census_income), ('german', pre_german_credit), ('bank', pre_bank_marketing)]:
    print(dataset_name + ':')
    for model_file, approach in [(dataset_name+'_model.h5', 'Original model'), (dataset_name+'_ADF_retrained_model.h5', 'Retrained with ADF'),
                                 (dataset_name+'_EIDIG_5_retrained_model.h5', 'Retrained with EIDIG-5'), (dataset_name+'_EIDIG_INF_retrained_model.h5', 'Retrained with EIDIG-INF')]:
        if approach == 'Original model':
            model = keras.models.load_model('models/original_models/' + model_file)
        else:
            model = keras.models.load_model('models/retrained_models/' + model_file)
        print(approach + ':')
        precision_recall(dataset_module.y_test, (model.predict(dataset_module.X_test)>0.5).astype(int).flatten())


"""
adult:
Original model:
The precision rate is  0.7338425381903643 , the recall rate is  0.5454148471615721 , and the F1 score is  0.625751503006012
Model retrained with ADF:
The precision rate is  0.7543988269794721 , the recall rate is  0.4493449781659389 , and the F1 score is  0.5632183908045977
Model retrained with EIDIG-5:
The precision rate is  0.7149505526468877 , the recall rate is  0.5366812227074236 , and the F1 score is  0.6131204789224245
Model retrained with EIDIG-INF:
The precision rate is  0.6909937888198758 , the recall rate is  0.5829694323144105 , and the F1 score is  0.6324017053529134

german:
Original model:
The precision rate is  0.6781609195402298 , the recall rate is  0.5 , and the F1 score is  0.5756097560975609
Model retrained with ADF:
The precision rate is  0.6588235294117647 , the recall rate is  0.4745762711864407 , and the F1 score is  0.5517241379310346
Model retrained with EIDIG-5:
The precision rate is  0.6413043478260869 , the recall rate is  0.5 , and the F1 score is  0.5619047619047619
Model retrained with EIDIG-INF:
The precision rate is  0.5655737704918032 , the recall rate is  0.5847457627118644 , and the F1 score is  0.575

bank:
Original model:
The precision rate is  0.7181467181467182 , the recall rate is  0.17048579285059579 , and the F1 score is  0.27555555555555555
Model retrained with ADF:
The precision rate is  0.6234413965087282 , the recall rate is  0.229147571035747 , and the F1 score is  0.3351206434316354
Model retrained with EIDIG-5:
The precision rate is  0.6543909348441926 , the recall rate is  0.21173235563703025 , and the F1 score is  0.31994459833795014
Model retrained with EIDIG-INF:
The precision rate is  0.6224489795918368 , the recall rate is  0.2236480293308891 , and the F1 score is  0.32906271072151044
"""
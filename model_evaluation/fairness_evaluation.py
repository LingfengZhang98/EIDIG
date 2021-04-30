"""
This python file evaluates the discriminatory degree of models.
"""


import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
 
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
from tensorflow import keras
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
import generation_utilities


def ids_percentage(sample_round, num_gen, num_attribs, protected_attribs, constraint, model):
    # compute the percentage of individual discriminatory instances with 95% confidence

    statistics = np.empty(shape=(0, ))
    for i in range(sample_round):
        gen_id = generation_utilities.purely_random(num_attribs, protected_attribs, constraint, model, num_gen)
        percentage = len(gen_id) / num_gen
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)


# load models
adult_model = keras.models.load_model("models/original_models/adult_model.h5")
german_model = keras.models.load_model("models/original_models/german_model.h5")
bank_model = keras.models.load_model("models/original_models/bank_model.h5")

adult_ADF_retrained_model = keras.models.load_model("models/retrained_models/adult_ADF_retrained_model.h5")
adult_EIDIG_5_retrained_model = keras.models.load_model("models/retrained_models/adult_EIDIG_5_retrained_model.h5")
adult_EIDIG_INF_retrained_model = keras.models.load_model("models/retrained_models/adult_EIDIG_INF_retrained_model.h5")

german_ADF_retrained_model = keras.models.load_model("models/retrained_models/german_ADF_retrained_model.h5")
german_EIDIG_5_retrained_model = keras.models.load_model("models/retrained_models/german_EIDIG_5_retrained_model.h5")
german_EIDIG_INF_retrained_model = keras.models.load_model("models/retrained_models/german_EIDIG_INF_retrained_model.h5")

bank_ADF_retrained_model = keras.models.load_model("models/retrained_models/bank_ADF_retrained_model.h5")
bank_EIDIG_5_retrained_model = keras.models.load_model("models/retrained_models/bank_EIDIG_5_retrained_model.h5")
bank_EIDIG_INF_retrained_model = keras.models.load_model("models/retrained_models/bank_EIDIG_INF_retrained_model.h5")


def measure_discrimination(sample_round, num_gen):
    # measure the discrimination degree of models on each benchmark

    print('Percentage of discriminatory instances for original model, model retrained with ADF, model retrained with EIDIG-5, and model retrained with EIDIG-IND, respectively:\n')

    for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
        print(benchmark, ':')
        ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]), protected_attribs, pre_census_income.constraint, adult_model)
        ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]), protected_attribs, pre_census_income.constraint, adult_ADF_retrained_model)
        ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]), protected_attribs, pre_census_income.constraint, adult_EIDIG_5_retrained_model)
        ids_percentage(sample_round, num_gen, len(pre_census_income.X[0]), protected_attribs, pre_census_income.constraint, adult_EIDIG_INF_retrained_model)
    
    for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
        print(benchmark, ':')
        ids_percentage(sample_round, num_gen, len(pre_german_credit.X[0]), protected_attribs, pre_german_credit.constraint, german_model)
        ids_percentage(sample_round, num_gen, len(pre_german_credit.X[0]), protected_attribs, pre_german_credit.constraint, german_ADF_retrained_model)
        ids_percentage(sample_round, num_gen, len(pre_german_credit.X[0]), protected_attribs, pre_german_credit.constraint, german_EIDIG_5_retrained_model)
        ids_percentage(sample_round, num_gen, len(pre_german_credit.X[0]), protected_attribs, pre_german_credit.constraint, german_EIDIG_INF_retrained_model)

    print('B-a:')
    ids_percentage(sample_round, num_gen, len(pre_bank_marketing.X[0]), [0], pre_bank_marketing.constraint, bank_model)
    ids_percentage(sample_round, num_gen, len(pre_bank_marketing.X[0]), [0], pre_bank_marketing.constraint, bank_ADF_retrained_model)
    ids_percentage(sample_round, num_gen, len(pre_bank_marketing.X[0]), [0], pre_bank_marketing.constraint, bank_EIDIG_5_retrained_model)
    ids_percentage(sample_round, num_gen, len(pre_bank_marketing.X[0]), [0], pre_bank_marketing.constraint, bank_EIDIG_INF_retrained_model)


# reproduce the results reported by our paper
# measure_discrimination(100, 10000)


# just for test
measure_discrimination(10, 100)
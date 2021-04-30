"""
This python file calls functions from experiments.py to reproduce the main experiments of our paper.
"""


import experiments
from preprocessing import pre_census_income
from preprocessing import pre_german_credit
from preprocessing import pre_bank_marketing
from tensorflow import keras
import numpy as np


"""
for census income data, age(0), race(6) and gender(7) are protected attributes in 12 features
for german credit data, gender(6) and age(9) are protected attributes in 24 features
for bank marketing data, age(0) is protected attribute in 16 features
"""


# load models
adult_model = keras.models.load_model("models/original_models/adult_model.h5")
german_model = keras.models.load_model("models/original_models/german_model.h5")
bank_model = keras.models.load_model("models/original_models/bank_model.h5")


# compare the global generation phase on each benchmark, and evaluate the effect of decay factor
ROUND = 5
decay_list = np.arange(0, 1.1, 0.1)
sum_num_ids = np.array([0] * (len(decay_list)+1))
sum_num_iter = np.array([0] * (len(decay_list)+1))
sum_time_cost = np.array([0] * (len(decay_list)+1))

for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
    print('\n', benchmark, ':\n')
    num_ids, num_iter, time_cost = experiments.global_comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, decay_list)
    sum_num_ids += num_ids
    sum_num_iter += num_iter
    sum_time_cost += time_cost
for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
    print('\n', benchmark, ':\n')
    num_ids, num_iter, time_cost = experiments.global_comparison(ROUND, benchmark, pre_german_credit.X_train, protected_attribs, pre_german_credit.constraint, german_model, decay_list)
    sum_num_ids += num_ids
    sum_num_iter += num_iter
    sum_time_cost += time_cost
print('\nB-a:\n')
num_ids, num_iter, time_cost = experiments.global_comparison(ROUND, 'B-a', pre_bank_marketing.X_train, [0], pre_bank_marketing.constraint, bank_model, decay_list)
sum_num_ids += num_ids
sum_num_iter += num_iter
sum_time_cost += time_cost

avg_num_ids = sum_num_ids / ROUND
avg_iter = sum_num_iter / ROUND / (7*1000+3*600)
avg_speed = sum_num_ids / sum_time_cost

print('Results of global phase comparsion, averaged on', ROUND, 'rounds:')
print('ADF:', avg_num_ids[0], 'individual discriminatory instances are generated given 1000 (600 for german credit) seeds for each benchmark at a speed of', avg_speed[0], 'per second, and the number of iterations on a singe seed is', avg_iter[0], '.')
for index, decay in enumerate(decay_list):
    print('Decay factor set to {}:'.format(decay))
    print('EIDIG:', avg_num_ids[index+1], 'individual discriminatory instances are generated given 1000 (600 for german credit) seeds for each benchmark at a speed of', avg_speed[index+1], 'per second, and the number of iterations on a singe seed is', avg_iter[index+1], '.')


# compare the local generation phase on each benchmark, and evaluate the effect of update interval
ROUND = 5
update_interval_list = np.append(np.arange(1, 11, 1), 10000)
sum_num_ids = np.array([0] * (len(update_interval_list)+1))
sum_time_cost = np.array([0] * (len(update_interval_list)+1))

for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
    print('\n', benchmark, ':\n')
    num_ids, time_cost = experiments.local_comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model, update_interval_list)
    sum_num_ids += num_ids
    sum_time_cost += time_cost
for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
    print('\n', benchmark, ':\n')
    num_ids, time_cost = experiments.local_comparison(ROUND, benchmark, pre_german_credit.X_train, protected_attribs, pre_german_credit.constraint, german_model, update_interval_list)
    sum_num_ids += num_ids
    sum_time_cost += time_cost
print('\nB-a:\n')
num_ids, time_cost = experiments.local_comparison(ROUND, 'B-a', pre_bank_marketing.X_train, [0], pre_bank_marketing.constraint, bank_model, update_interval_list)
sum_num_ids += num_ids
sum_time_cost += time_cost

avg_num_ids = sum_num_ids / ROUND
avg_speed = sum_num_ids / sum_time_cost

print('Results of local phase comparsion, averaged on', ROUND, 'rounds:')
print('ADF:', avg_num_ids[0], 'individual discriminatory instances are generated given 100 discriminatory seeds for each benchmark at a speed of', avg_speed[0], 'per second.')
for index, update_interval in enumerate(update_interval_list):
    print('Decay factor set to {}:'.format(update_interval))
    print('EIDIG:', avg_num_ids[index+1], 'individual discriminatory instances are generated given 100 discriminatory seeds for each benchmark at a speed of', avg_speed[index+1], 'per second.')


# complete comparison on each benchmark
ROUND = 5

for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
    print('\n', benchmark, ':\n')
    experiments.comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model)
for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
    print('\n', benchmark, ':\n')
    experiments.comparison(ROUND, benchmark, pre_german_credit.X_train, protected_attribs, pre_german_credit.constraint, german_model)
print('\nB-a:\n')
experiments.comparison(ROUND, 'B-a', pre_bank_marketing.X_train, [0], pre_bank_marketing.constraint, bank_model)


# compare effectiveness in a seedwise fashion
ROUND = 5

for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
    print('\n', benchmark, ':\n')
    experiments.seedwise_comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model)
for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
    print('\n', benchmark, ':\n')
    experiments.seedwise_comparison(ROUND, benchmark, pre_german_credit.X_train, protected_attribs, pre_german_credit.constraint, german_model)
print('\nB-a:\n')
experiments.seedwise_comparison(ROUND, 'B-a', pre_bank_marketing.X_train, [0], pre_bank_marketing.constraint, bank_model)


# time consumption comparison for generating a certain number of discriminatory instances
ROUND = 5

for benchmark, protected_attribs in [('C-a', [0]), ('C-r', [6]), ('C-g', [7]), ('C-a&r', [0,6]), ('C-a&g', [0,7]), ('C-r&g', [6,7])]:
    print('\n', benchmark, ':\n')
    experiments.time_cost_comparison(ROUND, benchmark, pre_census_income.X_train, protected_attribs, pre_census_income.constraint, adult_model)
for benchmark, protected_attribs in [('G-g', [6]), ('G-a', [9]), ('G-g&a', [6,9])]:
    print('\n', benchmark, ':\n')
    experiments.time_cost_comparison(ROUND, benchmark, pre_german_credit.X_train, protected_attribs, pre_german_credit.constraint, german_model)
print('\nB-a:\n')
experiments.time_cost_comparison(ROUND, 'B-a', pre_bank_marketing.X_train, [0], pre_bank_marketing.constraint, bank_model)
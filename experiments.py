"""
This python file provides experimental functions backing up the claims involving efficiency and effectiveness in our paper.
"""


import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import generation_utilities
import time
import ADF
import EIDIG


# allocate GPU and set dynamic memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# make outputs stable across runs for validation
# alternatively remove them when dealing with real-world issues
np.random.seed(42)
tf.random.set_seed(42)


def comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, g_num=1000, l_num=1000, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare EIDIG with ADF in terms of effectiveness and efficiency

    num_ids = np.array([0] * 3)
    time_cost = np.array([0] * 3)

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        if g_num >= len(X):
            seeds = X.copy()
        else:
            clustered_data = generation_utilities.clustering(X, c_num)
            seeds = np.empty(shape=(0, len(X[0])))
            for i in range(g_num):
                new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion=fashion)
                seeds = np.append(seeds, [new_seed], axis=0)

        t1 = time.time()
        ids_ADF, gen_ADF, total_iter_ADF = ADF.individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, l_num, max_iter, s_g, s_l, epsilon_l)
        np.save('logging_data/logging_data_from_tests/complete_comparison/' + benchmark + '_ids_ADF_' + str(round_now) + '.npy', ids_ADF)
        t2 = time.time()
        print('ADF:', 'In', total_iter_ADF, 'search iterations', len(gen_ADF), 'non-duplicate instances are explored', len(ids_ADF), 'of which are discriminatory. Time cost:', t2-t1, 's.')
        num_ids[0] += len(ids_ADF)
        time_cost[0] += t2-t1

        t1 = time.time()
        ids_EIDIG_5, gen_EIDIG_5, total_iter_EIDIG_5 = EIDIG.individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, decay, l_num, 5, max_iter, s_g, s_l, epsilon_l)
        np.save('logging_data/logging_data_from_tests/complete_comparison/' + benchmark + '_ids_EIDIG_5_' + str(round_now) + '.npy', ids_EIDIG_5)
        t2 = time.time()
        print('EIDIG-5:', 'In', total_iter_EIDIG_5, 'search iterations', len(gen_EIDIG_5), 'non-duplicate instances are explored', len(ids_EIDIG_5), 'of which are discriminatory. Time cost:', t2-t1, 's.')
        num_ids[1] += len(ids_EIDIG_5)
        time_cost[1] += t2-t1

        t1 = time.time()
        ids_EIDIG_INF, gen_EIDIG_INF, total_iter_EIDIG_INF = EIDIG.individual_discrimination_generation(X, seeds, protected_attribs, constraint, model, decay, l_num, l_num+1, max_iter, s_g, s_l, epsilon_l)
        np.save('logging_data/logging_data_from_tests/complete_comparison/' + benchmark + '_ids_EIDIG_INF_' + str(round_now) + '.npy', ids_EIDIG_INF)
        t2 = time.time()
        print('EIDIG-INF:', 'In', total_iter_EIDIG_INF, 'search iterations', len(gen_EIDIG_INF), 'non-duplicate instances are explored', len(ids_EIDIG_INF), 'of which are discriminatory. Time cost:', t2-t1, 's.')
        num_ids[2] += len(ids_EIDIG_INF)
        time_cost[2] += t2-t1

        print('\n')
    
    avg_num_ids = num_ids / num_experiment_round
    avg_speed = num_ids / time_cost
    print('Results of complete comparison on', benchmark, 'with g_num set to {} and l_num set to {}'.format(g_num, l_num), ',averaged on', num_experiment_round, 'rounds:')
    for index, approach in [(0, 'ADF'), (1, 'EIDIG-5'), (2, 'EIDIG-INF')]:
        print(approach, ':', avg_num_ids[index], 'individual discriminatory instances are generated at a speed of', avg_speed[index], 'per second.')


def global_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, decay_list, num_seeds=1000, c_num=4, max_iter=10, s_g=1.0):
    # compare the global phase given the same set of seeds

    num_ids = np.array([0] * (len(decay_list) + 1))
    num_iter = np.array([0] * (len(decay_list) + 1))
    time_cost = np.array([0] * (len(decay_list) + 1))

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        num_attribs = len(X[0])
        num_dis = 0
        if num_seeds >= len(X):
            seeds = X
        else:
            clustered_data = generation_utilities.clustering(X, c_num)
            seeds = np.empty(shape=(0, num_attribs))
            for i in range(num_seeds):
                x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion='Distribution')
                seeds = np.append(seeds, [x_seed], axis=0)
        for seed in seeds:
            similar_seed = generation_utilities.similar_set(seed, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(seed, similar_seed, model):
                num_dis += 1
        print('Given', num_seeds, '(no more than 600 for german credit) seeds,', num_dis, 'of which are individual discriminatory instances.')

        t1 = time.time()
        ids_ADF, _, total_iter_ADF = ADF.global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, max_iter, s_g)
        t2 = time.time()
        num_ids_ADF = len(ids_ADF)
        print('ADF:', 'In', total_iter_ADF, 'search iterations,', num_ids_ADF, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
        num_ids[0] += num_ids_ADF
        num_iter[0] += total_iter_ADF
        time_cost[0] += t2-t1

        for index, decay in enumerate(decay_list):
            print('Decay factor set to {}:'.format(decay))
            t1 = time.time()
            ids_EIDIG, _, total_iter_EIDIG = EIDIG.global_generation(X, seeds, num_attribs, protected_attribs, constraint, model, decay, max_iter, s_g)
            t2 = time.time()
            num_ids_EIDIG = len(ids_EIDIG)
            print('EIDIG:', 'In', total_iter_EIDIG, 'search iterations,', num_ids_EIDIG, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
            num_ids[index+1] += num_ids_EIDIG
            num_iter[index+1] += total_iter_EIDIG
            time_cost[index+1] += t2-t1
        
        print('\n')

    avg_num_ids = num_ids / num_experiment_round
    avg_speed = num_ids / time_cost
    avg_iter = num_iter / num_experiment_round / num_seeds
    print('Results of global phase comparsion on', benchmark, 'given {} seeds'.format(num_seeds), ',averaged on', num_experiment_round, 'rounds:')
    print('ADF:', avg_num_ids[0], 'individual discriminatory instances are generated at a speed of', avg_speed[0], 'per second, and the number of iterations on a singe seed is', avg_iter[0], '.')
    for index, decay in enumerate(decay_list):
        print('Decay factor set to {}:'.format(decay))
        print('EIDIG:', avg_num_ids[index+1], 'individual discriminatory instances are generated at a speed of', avg_speed[index+1], 'per second, and the number of iterations on a singe seed is', avg_iter[index+1], '.')

    return num_ids, num_iter, time_cost
    

def local_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, update_interval_list, num_seeds=100, l_num=1000, c_num=4, s_l=1.0, epsilon=1e-6):
    # compare the local phase given the same individual discriminatory instances set

    num_ids = np.array([0] * (len(update_interval_list) + 1))
    time_cost = np.array([0] * (len(update_interval_list) + 1))

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        num_attribs = len(X[0])
        clustered_data = generation_utilities.clustering(X, c_num)
        id_seeds = np.empty(shape=(0, num_attribs))
        for i in range(100000000):
            x_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion='RoundRobin')
            similar_x_seed = generation_utilities.similar_set(x_seed, num_attribs, protected_attribs, constraint)
            if generation_utilities.is_discriminatory(x_seed, similar_x_seed, model):
                id_seeds = np.append(id_seeds, [x_seed], axis=0)
                if len(id_seeds) >= num_seeds:
                    break
    
        t1 = time.time()
        ids_ADF, _, total_iter_ADF = ADF.local_generation(num_attribs, l_num, id_seeds.copy(), protected_attribs, constraint, model, s_l, epsilon)
        t2 = time.time()
        num_ids_ADF = len(ids_ADF)
        print('ADF:', 'In', total_iter_ADF, 'search iterations,', num_ids_ADF, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
        num_ids[0] += num_ids_ADF
        time_cost[0] += t2-t1
        
        for index, update_interval in enumerate(update_interval_list):
            print('Update interval set to {}:'.format(update_interval))
            t1 = time.time()
            ids_EIDIG, _, total_iter_EIDIG = EIDIG.local_generation(num_attribs, l_num, id_seeds.copy(), protected_attribs, constraint, model, update_interval, s_l, epsilon)
            t2 = time.time()
            num_ids_EIDIG = len(ids_EIDIG)
            print('EIDIG:', 'In', total_iter_EIDIG, 'search iterations,', num_ids_EIDIG, 'non-duplicate individual discriminatory instances are generated. Time cost:', t2-t1, 's.')
            num_ids[index+1] += num_ids_EIDIG
            time_cost[index+1] += t2-t1
        
        print('\n')

    avg_num_ids = num_ids / num_experiment_round
    avg_speed = num_ids / time_cost
    print('Results of local phase comparsion on', benchmark, 'with l_num set to {} given {} discriminatory seeds'.format(l_num, num_seeds), ',averaged on', num_experiment_round, 'rounds:')
    print('ADF:', avg_num_ids[0], 'individual discriminatory instances are generated at a speed of', avg_speed[0], 'per second.')
    for index, update_interval in enumerate(update_interval_list):
        print('Update interval set to {}:'.format(update_interval))
        print('EIDIG:', avg_num_ids[index+1], 'individual discriminatory instances are generated at a speed of', avg_speed[index+1], 'per second.')
    
    return num_ids, time_cost


def seedwise_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, g_num=100, l_num=100, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare the number of non-duplicate individual discriminatory instances generated in a seedwise fashion

    num_gen = np.zeros([3, g_num])
    num_ids = np.zeros([3, g_num])

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        clustered_data = generation_utilities.clustering(X, c_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(g_num):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion=fashion)
            seeds = np.append(seeds, [new_seed], axis=0)

        gen_ADF, ids_ADF = ADF.seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, max_iter, s_g, s_l, epsilon_l)
        gen_EIDIG_5, ids_EIDIG_5 = EIDIG.seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, 0.5, 5, max_iter, s_g, s_l, epsilon_l)
        gen_EIDIG_INF, ids_EIDIG_INF = EIDIG.seedwise_generation(X, seeds, protected_attribs, constraint, model, l_num, 0.5, l_num+1, max_iter, s_g, s_l, epsilon_l)
        num_gen[0] += gen_ADF
        num_ids[0] += ids_ADF
        num_gen[1] += gen_EIDIG_5
        num_ids[1] += ids_EIDIG_5
        num_gen[2] += gen_EIDIG_INF
        num_ids[2] += ids_EIDIG_INF
    
    avg_num_gen = num_gen / num_experiment_round
    avg_num_ids = num_ids / num_experiment_round
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_gen_ADF.npy', num_gen[0])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_ids_ADF.npy', num_ids[0])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_gen_EIDIG_5.npy', num_gen[1])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_ids_EIDIG_5.npy', num_ids[1])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_gen_EIDIG_INF.npy', num_gen[2])
    np.save('logging_data/logging_data_from_tests/seedwise_comparison/' + benchmark + '_num_ids_EIDIG_INF.npy', num_ids[2])

    print('Results averaged on', num_experiment_round, 'rounds have been saved. Results on the first 10 seeds:')
    print('ADF:')
    print('Number of generated instances:', num_gen[0, :10])
    print('Number of generated discriminatory instances:', num_ids[0, :10])
    print('EIDIG-5:')
    print('Number of generated instances:', num_gen[1, :10])
    print('Number of generated discriminatory instances:', num_ids[1, :10])
    print('EIDIG-INF:')
    print('Number of generated instances:', num_gen[2, :10])
    print('Number of generated discriminatory instances:', num_ids[2, :10])


def time_cost_comparison(num_experiment_round, benchmark, X, protected_attribs, constraint, model, record_step=100, record_frequency=100, g_num=1000, l_num=1000, decay=0.5, c_num=4, max_iter=10, s_g=1.0, s_l=1.0, epsilon_l=1e-6, fashion='RoundRobin'):
    # compare the time consumption for generating a certain number of non-duplicate individual discriminatory instances

    time_cost = np.zeros([3, record_frequency])

    for i in range(num_experiment_round):
        round_now = i + 1
        print('--- ROUND', round_now, '---')
        if len(X) <= g_num:
            seeds = X.copy()
        else:
            clustered_data = generation_utilities.clustering(X, c_num)
            seeds = np.empty(shape=(0, len(X[0])))
            for i in range(g_num):
                new_seed = generation_utilities.get_seed(clustered_data, len(X), c_num, i%c_num, fashion=fashion)
                seeds = np.append(seeds, [new_seed], axis=0)

        t_ADF = ADF.time_record(X, seeds, protected_attribs, constraint, model, l_num, record_step, record_frequency, max_iter, s_g, s_l, epsilon_l)
        t_EIDIG_5 = EIDIG.time_record(X, seeds, protected_attribs, constraint, model, decay, l_num, record_step, record_frequency, 5, max_iter, s_g, s_l, epsilon_l)
        t_EIDIG_INF = EIDIG.time_record(X, seeds, protected_attribs, constraint, model, decay, l_num, record_step, record_frequency, l_num+1, max_iter, s_g, s_l, epsilon_l)
        time_cost[0] += t_ADF
        time_cost[1] += t_EIDIG_5
        time_cost[2] += t_EIDIG_INF

    avg_time_cost = time_cost / num_experiment_round
    np.save('logging_data/logging_data_from_tests/time_cost_comparison/' + benchmark + '_time_every{}ids_ADF.npy'.format(record_step), avg_time_cost[0])
    np.save('logging_data/logging_data_from_tests/time_cost_comparison/' + benchmark + '_time_every{}ids_EIDIG_5.npy'.format(record_step), avg_time_cost[1])
    np.save('logging_data/logging_data_from_tests/time_cost_comparison/' + benchmark + '_time_every{}ids_EIDIG_INF.npy'.format(record_step), avg_time_cost[2])

    print('Results averaged on', num_experiment_round, 'rounds have been saved. Results on the first 10 records:')
    print('ADF:')
    print('Time cost:', avg_time_cost[0, :10])
    print('EIDIG-5:')
    print('Time cost:', avg_time_cost[1, :10])
    print('EIDIG-INF:')
    print('Time cost:', avg_time_cost[2, :10])
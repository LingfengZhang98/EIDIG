"""
This python file provides essential functions for individual discrimination generation.
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import cluster
import itertools
import time


def clustering(data, c_num):
    # standard KMeans algorithm

    kmeans = cluster.KMeans(n_clusters=c_num)
    y_pred = kmeans.fit_predict(data)
    return [data[y_pred==n] for n in range(c_num)]


def clip(instance, constraint):
    # clip the generated instance to satisfy the constraint

    return np.minimum(constraint[:, 1], np.maximum(constraint[:, 0], instance))


def random_pick(probability):
    # randomly pick an element from a probability distribution

    random_number = np.random.rand()
    current_proba = 0
    for i in range(len(probability)):
        current_proba += probability[i]
        if current_proba > random_number:
            return i


def get_seed(clustered_data, X_len, c_num, cluster_i, fashion='RoundRobin'):
    # get a seed from the specified cluster in a round-robin fashion
    # alternatively choose 'Distribution' to randomly sample a seed from a cluster with the probability proportional to the cluster size

    if fashion == 'RoundRobin':
        index = np.random.randint(0, len(clustered_data[cluster_i]))
        return clustered_data[cluster_i][index]
    elif fashion == 'Distribution':
        pick_probability = [len(clustered_data[i]) / X_len for i in range(c_num)]
        x = clustered_data[random_pick(pick_probability)]
        index = np.random.randint(0, len(x))
        return x[index]


def similar_set(x, num_attribs, protected_attribs, constraint):
    # find all similar inputs corresponding to different combinations of protected attributes with non-protected attributes unchanged

    similar_x = np.empty(shape=(0, num_attribs))
    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(constraint[i][0], constraint[i][1]+1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for comb in all_combs:
        x_new = x.copy()
        for a, c in zip(protected_attribs, comb):
            x_new[a] = c
        similar_x = np.append(similar_x, [x_new], axis=0)
    return similar_x


def is_discriminatory(x, similar_x, model):
    # identify whether the instance is discriminatory w.r.t. the model
    y_pred = (model(tf.constant([x])) > 0.5)
    for x_new in similar_x:
        if (model(tf.constant([x_new])) > 0.5) != y_pred:
            return True
    return False


def max_diff(x, similar_x, model):
    # select a similar instance such that the DNN outputs on them are maximally different

    y_pred_proba = model(tf.constant([x]))
    def distance(x_new):
        return np.sum(np.square(y_pred_proba - model(tf.constant([x_new]))))
    max_dist = 0.0
    x_potential_pair = x.copy()
    for x_new in similar_x:
        if distance(x_new) > max_dist:
            max_dist = distance(x_new)
            x_potential_pair = x_new.copy()
    return x_potential_pair


def find_pair(x, similar_x, model):
    # find a discriminatory pair given an individual discriminatory instance

    pairs = np.empty(shape=(0, len(x)))
    y_pred = (model(tf.constant([x])) > 0.5)
    for x_pair in similar_x:
        if (model(tf.constant([x_pair])) > 0.5) != y_pred:
            pairs = np.append(pairs, [x_pair], axis=0)
    selected_p = random_pick([1.0 / pairs.shape[0]] * pairs.shape[0])
    return pairs[selected_p]


def normalization(grad1, grad2, protected_attribs, epsilon):
    # gradient normalization during local search

    gradient = np.zeros_like(grad1)
    grad1 = np.abs(grad1)
    grad2 = np.abs(grad2)
    for i in range(len(gradient)):
        saliency = grad1[i] + grad2[i]
        gradient[i] = 1.0 / (saliency + epsilon)
        if i in protected_attribs:
            gradient[i] = 0.0
    gradient_sum = np.sum(gradient)
    probability = gradient / gradient_sum
    return probability
    

def purely_random(num_attribs, protected_attribs, constraint, model, gen_num):
    # generate instances in a purely random fashion
    
    gen_id = np.empty(shape=(0, num_attribs))
    for i in range(gen_num):
        x_picked = [0] * num_attribs
        for a in range(num_attribs):
            x_picked[a] = np.random.randint(constraint[a][0], constraint[a][1]+1)
        if is_discriminatory(x_picked, similar_set(x_picked, num_attribs, protected_attribs, constraint), model):
            gen_id = np.append(gen_id, [x_picked], axis=0)
    return gen_id
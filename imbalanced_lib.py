import pandas as pd
from imblearn.under_sampling import NeighbourhoodCleaningRule, RandomUnderSampler
from keras import backend as K
import tensorflow as tf



def get_sampler(sampler_name):
    if sampler_name == 'srs':
        return srs_sampler
    elif sampler_name == 'ncr':
        return ncr_sampler
    else:
        return none_sampler



def none_sampler(data):
    return data



def srs_sampler(data):
    """
    Simple random sampling. Balances classes based on minority class cardinality.
    :param data: dataset with 'label' column
    :return: balanced dataset
    """
    srs = RandomUnderSampler(sampling_strategy='majority', random_state=7810)
    labels = data['label']
    data.drop(columns=['label'], inplace=True)
    data, y = srs.fit_resample(data, labels)
    data['label'] = y
    
    return data


def ncr_sampler(data):
    """
    Uses neighboor cleanning rule sampler to undersample majority class
    :param data: dataset with 'label' column
    :return: balanced dataset
    """
    
    ncr = NeighbourhoodCleaningRule(sampling_strategy='majority', n_jobs=24)
    labels = data['label']
    data.drop(columns=['label'], inplace=True)
    data, y = ncr.fit_resample(data, labels)
    data['label'] = y

    return data



def get_loss(loss_name):
    if loss_name == 'categorical':
        return 'categorical_crossentropy'
    elif loss_name == 'focal':
        return focal_loss()
    else:
        return None
    


def focal_loss(gamma=2., alpha=.25):
    """
    implementation of focal loss function for neural network training
    :return: focal loss callable to be used in keras
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed
    


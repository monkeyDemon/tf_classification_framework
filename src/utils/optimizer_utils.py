# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:05:58 2019

get optimizer

@author: as
"""
import tensorflow as tf

def get_optimizer(config_dict, lr_placeholder):
    optimizer_str = config_dict['SOLVER']['OPTIMIZER']
    if optimizer_str == 'momentum':
        momentum = config_dict['SOLVER']['MOMENTUM']
        print("use momentum sgd optimizer, momentum = {}".format(momentum))
        optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=momentum)
    elif optimizer_str == 'adam':
        print("use adam optimizer")
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_placeholder)
    else:
        raise RunTimeError("unknown optimizer: {}".format(optimizer_str))
    return optimizer

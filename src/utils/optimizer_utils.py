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
        momentum = config_dict['SOLVER']['OPTIMIZER_MOMENTUM']['MOMENTUM']
        use_nesterov = config_dict['SOLVER']['OPTIMIZER_MOMENTUM']['USE_NESTEROV']
        use_nesterov = True if use_nesterov == 1 else False
        print("use momentum sgd optimizer, momentum = {}, use_nesterov = {}".format(momentum, use_nesterov))
        optimizer = tf.compat.v1.train.MomentumOptimizer(lr_placeholder, momentum=momentum, use_nesterov=use_nesterov)
    elif optimizer_str == 'adam':
        beta1 = config_dict['SOLVER']['OPTIMIZER_ADAM']['BETA1']
        beta2 = config_dict['SOLVER']['OPTIMIZER_ADAM']['BETA2']
        print("use adam optimizer, beta1 = {}, beta2 = {}".format(beta1, beta2))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_placeholder, beta1=beta1, beta2=beta2)
    else:
        raise RunTimeError("unknown optimizer: {}".format(optimizer_str))
    return optimizer

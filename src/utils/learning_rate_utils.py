# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:05:58 2019

Learning rate Scheduler

@author: as
"""
from __future__ import division
import math
import numpy as np


def get_learning_rate_scheduler(config_dict):
    lr_policy = config_dict['SOLVER']['LR_POLICY']
    if lr_policy == 'steps_decay':
        schduler = Scheduler_Steps_Decay(config_dict)
        return schduler
    elif lr_policy == 'noise_filter':
        schduler = Scheduler_Noise_Filter(config_dict)
        return schduler
    elif lr_policy == 'cosine_decay':
        schduler = Scheduler_Cosine_Decay(config_dict)
        return schduler
    else:
        raise RunTimeError("unknown lr policy: {}".format(lr_policy))



class Scheduler_Steps_Decay(object):
    """Steps decay learning rate Scheduler"""

    def __init__(self, config_dict):
        """Constructor."""
        self.base_lr = config_dict['SOLVER']['POLICY_STEPS_DECAY']['BASE_LR']
        self.lr_decay_factor = config_dict['SOLVER']['POLICY_STEPS_DECAY']['LR_DECAY_FACTOR']
        self.epochs_per_decay = config_dict['SOLVER']['POLICY_STEPS_DECAY']['EPOCHS_PER_DECAY']

    def get_learning_rate(self, epoch, batch_idx, batch_num_per_epoch):
        decay_cnt = epoch // self.epochs_per_decay
        lr = self.base_lr
        for idx in range(decay_cnt):
            lr *= self.lr_decay_factor
        return lr
            

class Scheduler_Noise_Filter(object):
    """Learning rate Scheduler for matching noise labels filter."""

    def __init__(self, config_dict):
        """Constructor"""
        self.base_lr = config_dict['SOLVER']['POLICY_NOISE_FILTER']['BASE_LR']
        self.fixed_epoch = config_dict['SOLVER']['POLICY_NOISE_FILTER']['FIXED_EPOCH']
        self.max_lr = config_dict['SOLVER']['POLICY_NOISE_FILTER']['MAX_LR']
        self.min_lr = config_dict['SOLVER']['POLICY_NOISE_FILTER']['MIN_LR']

    def get_learning_rate(self, epoch, batch_idx, batch_num_per_epoch):
        if epoch < self.fixed_epoch:
            return self.base_lr
        lr = self.max_lr - (self.max_lr - self.min_lr) * (batch_idx / batch_num_per_epoch)
        return lr
        

class Scheduler_Cosine_Decay(object):
    """Cosine decay learning rate Scheduler"""

    def __init__(self, config_dict):
        """ Constructor"""
        self.base_lr = config_dict['SOLVER']['POLICY_COSINE_DECAY']['BASE_LR']
        self.total_epochs = config_dict['SOLVER']['MAX_EPOCH_NUMS']

    def get_learning_rate(self, epoch, batch_idx, batch_num_per_epoch):
        t_total = self.total_epochs * batch_num_per_epoch
        t_cur = float(epoch * batch_num_per_epoch + batch_idx)
        return 0.5 * self.base_lr * (1 + np.cos(np.pi * t_cur / t_total))


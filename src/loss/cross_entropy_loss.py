# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:25:05 2019

cross entropy loss function

@author: as
"""
import tensorflow as tf


def cross_entropy_loss(logits, labels, label_smooth=0.1):
    # logits: 模型输出未经处理的结果，如[2.45, 0.42, 1.28]
    # labels: one-hot 编码的 groundtruth 标签
    
    losses = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=labels,
        label_smoothing=label_smooth,
        reduction=tf.losses.Reduction.NONE)
    
    #loss = tf.reduce_mean(losses)
    #return loss
    return losses

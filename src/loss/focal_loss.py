# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:25:05 2019

focal loss function

@author: as
"""
import tensorflow as tf


def focal_loss(logits, labels):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Keyword Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls], datatype=float32
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls], datatype=float32
        gamma {float} -- (default: {2.0})
        alpha {constant tensor} -- each value respect to each category's weight

    Returns:
        A dictionary mapping strings (loss names) to loss values.
    """
    epsilon = 1.e-9
    gamma = 1.0  # 2.0
    alpha = tf.constant([[1], [2]], dtype=tf.float32)

    y_pred = tf.nn.softmax(logits, name='softmax_for_focalloss')

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(labels, -tf.log(model_out))
    weight = tf.multiply(labels, tf.pow(tf.subtract(1., model_out), gamma))
    #fl = tf.multiply(alpha, tf.multiply(weight, ce))
    #reduced_fl = tf.reduce_max(fl, axis=1)
    fl = tf.matmul(tf.multiply(weight, ce), alpha)
    loss = tf.reduce_mean(fl)
    return loss


# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 21:25:05 2019

common function

@author: as
"""
import tensorflow as tf


def compute_accuracy(logits, labels):
    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy

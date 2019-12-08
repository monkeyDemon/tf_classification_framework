# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:35:33 2019

build resnet_v1 model

you can build some kinds of official resnet v1 models with differents layer numbers.
e.g. resnet50, resnet101, resnet152, resnet200
these models can be fine tune on the ImageNet pretrained weights.

and for lightweight scenarios,
you can also customize some resnet v1 model with shallower layers.
e.g. resnet20

this file is modified on tensorflow/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py
if you use this file, you need to follow the original author's license.

@author: as
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

import resnet_v1
from loss.common import compute_accuracy


class ResNetModel(object):
    """ResNet_v1 Model builder."""
    
    def __init__(self, config_dict):
        """
        Constructor.
        Args:
            config_dict: a dictionary that saves the parameters config info 
        """
        self.network_name = config_dict['MODEL']['NETWORK_NAME']
        self.num_classes = config_dict['MODEL']['NUM_CLASSES']
        self.loss_name = config_dict['TRAIN']['LOSS']
        self.label_smooth = config_dict['TRICKS']['LABEL_SMOOTH_EPSILON']
        self.regular_loss_weight = config_dict['TRAIN']['REGULAR_LOSS_WEIGHT']
        
        if self.network_name not in resnet_v1.resnet_family:
            raise RuntimeError("error use of parameter: {}".format(self.network_name))
        
        # get resnet build function by network_name
        self.resnet_build_func = self._get_resnet_build_func()
    
    
    @property
    def get_num_classes(self):
        return self.num_classes

    @property
    def logits_op(self):
        return self.logits

    @property
    def predict_op(self):
        return self.predictions

    @property
    def accuracy_op(self):
        return self.accuracy

    @property
    def loss_op(self):
        return self.loss

    @property
    def predicted_loss_op(self):
        return self.predict_loss

    @property
    def regularized_loss_op(self):
        return self.regular_loss
    

    def _get_resnet_build_func(self):
        if self.network_name == 'resnet-v1-20':
            resnet_build_func = resnet_v1.resnet_v1_20
        elif self.network_name == 'resnet-v1-35':
            resnet_build_func = resnet_v1.resnet_v1_35
        elif self.network_name == 'resnet-v1-50':
            resnet_build_func = resnet_v1.resnet_v1_50
        elif self.network_name == 'resnet-v1-101':
            resnet_build_func = resnet_v1.resnet_v1_101
        elif self.network_name == 'resnet-v1-152':
            resnet_build_func = resnet_v1.resnet_v1_152
        elif self.network_name == 'resnet-v1-200':
            resnet_build_func = resnet_v1.resnet_v1_200
        else:
            raise RuntimeError("unknow network_name: {}".format(self.network_name))
        return resnet_build_func  
    
    
    def build_model(self, inputs, labels, is_training, reuse=False):
        """ Graph Input """

        # preprocess
        preprocessed_inputs = self.preprocess(inputs)
        
        # inference
        self.logits = self.inference(preprocessed_inputs, is_training=is_training, reuse=reuse)

        # prediction (this op is important when final use, record the op_name)
        self.predictions = tf.nn.softmax(self.logits, name='score_list')
        
        # accuracy
        self.accuracy = compute_accuracy(logits=self.logits, labels=labels)
        
        # loss (current options: cross_entropy_loss, focal_loss)    
        if self.loss_name == 'cross_entropy':
            from loss.cross_entropy_loss import cross_entropy_loss
            self.predict_loss = cross_entropy_loss(logits=self.logits, labels=labels, label_smooth=self.label_smooth)
        elif self.loss_name == 'focal_loss':
            from loss.focal_loss import focal_loss
            self.predict_loss = focal_loss(logits=self.logits, labels=labels) 
        else:
            raise RuntimeError("Error! not support loss_name: {} now".format(self.loss_name))

        self.regular_loss = tf.losses.get_regularization_loss()
        self.loss = tf.reduce_mean(self.predict_loss) + self.regular_loss_weight * self.regular_loss

        # show model structure
        self.show_all_variables()


    def preprocess(self, inputs):
        ''' preprocess
        here, we assuming that the size of the input image is correct 
        '''
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128)
        preprocessed_inputs = tf.div(preprocessed_inputs, 128)
        return preprocessed_inputs
    

    def inference(self, x, is_training=True, reuse=False):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, endpoints = self.resnet_build_func(
                inputs=x,
                num_classes=None,
                is_training=is_training,
                reuse=reuse,
                scope=self.network_name)
        
        with tf.variable_scope('Logits'):
            # the last average pooling layer makes the resnet50 ouput tensor with shape [None, 1, 1, 2048]
            # use tf.squeeze to flatten it into [None, 2048]
            net = tf.squeeze(net, axis=[1, 2])
            #net = slim.fully_connected(net, 512, scope='fc_inter')
            #net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='fc_dropout')
            logits = slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, scope='fc')
        return logits


    def show_all_variables(self):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)


    def get_vars_to_restore(self, train_mode):
        if train_mode == 'finetune_on_imagenet':
            # if train from imagenet pretrained model, exclude the last classification layer
            checkpoint_exclude_scopes = 'Logits'
            exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        elif train_mode == 'continue_training':
            # if continue training, just restore the whole model
            exclusions = []
        else:
            raise RuntimeError("train_mode error")

        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                variables_to_restore.append(var)
    
        return variables_to_restore

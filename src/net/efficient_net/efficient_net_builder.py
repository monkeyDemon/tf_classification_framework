# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:35:33 2019

build efficient-net model

this core code in official_efficientnet is modified on 
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
if you use this file, you need to follow the original tensorflow author's license.

@author: as
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops


from net.efficient_net.official_efficientnet import efficientnet_builder
from loss.common import compute_accuracy


efficient_net_family = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
    'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6']


class EfficientNetModel(object):
    """ResNet_v1 Model builder."""
    
    def __init__(self, config_dict):
        """Constructor.
        
        Args:
            network_name: structure name of network. e.g: efficientnet-b0
            num_classes: Number of classes.
        """
        self.network_name = config_dict['MODEL']['NETWORK_NAME']
        self.num_classes = config_dict['MODEL']['NUM_CLASSES']
        self.loss_name = config_dict['TRAIN']['LOSS']
        self.label_smooth = config_dict['TRICKS']['LABEL_SMOOTH_EPSILON']
        self.regular_loss_weight = config_dict['TRAIN']['REGULAR_LOSS_WEIGHT']
        
        if self.network_name not in efficient_net_family:
            raise RuntimeError("error use of parameter: {}".format(self.network_name))
        
        # get resnet build function by network_name
        #self.efficient_net_build_func = self._get_resnet_build_func()
        self.efficient_net_build_func = efficientnet_builder.build_model_base 
   
    
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
        # get EfficientNet's Feature Extractor
        features, endpoints = self.efficient_net_build_func(x, self.network_name, training=is_training) 
        
        # add fc layer
        with tf.variable_scope('Logits'):
            # Global average pooling.  e.g. from [None, 7, 7, 448] to [None, 1, 1, 448]
            net = math_ops.reduce_mean(features, [1, 2], name='pool5', keepdims=True)
            # use tf.squeeze to flatten [None, 1, 1, 448] into [None, 448]
            net = tf.squeeze(net, axis=[1, 2])
            # dropout
            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='scope')
            # fc layer
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

        # get total trainable vars
        variables_list = tf.trainable_variables()

        # remove exclusion vars
        variables_to_restore = [] 
        for var in variables_list:
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                variables_to_restore.append(var)

        return variables_to_restore



    def get_ema_vars(self):
      """Get all exponential moving average (ema) variables."""
      ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
      for v in tf.global_variables():
        # We maintain mva for batch norm moving mean and variance as well.
        if 'moving_mean' in v.name or 'moving_variance' in v.name:
          ema_vars.append(v)
      return list(set(ema_vars))

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:49:09 2018

@author: as
"""
import os
import tensorflow as tf

# Note: We need to import addditional module to fix the following bug:
# tensorflow.python.framework.errors_impl.NotFoundError: Op type not 
# registered 'ImageProjectiveTransform' in binary running on BJGS-SF-81. 
# Make sure the Op and Kernel are registered in the binary running in this 
# process. Note that if you are loading a saved graph which used ops from 
# tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done before 
# importing the graph, as contrib ops are lazily registered when the module 
# is first accessed.
import tensorflow.contrib.image

#from net.efficient_net.efficient_net_ckpt_predictor import EfficientNet_Predictor
#from utils.config_utils import import_ckpt_predictor_by_networkname
from utils.config_utils import import_model_by_networkname
#from efficient_net_builder import EfficientNetModel as Model



def get_predictor(weight_path, config_dict):
    network_name = config_dict['MODEL']['NETWORK_NAME']

    # import model by network_name
    import_str = import_model_by_networkname(network_name)
    exec(import_str)

    #predictor = None
    #if network_name.startswith('efficientnet'):
    #    enable_ema = True if paras['use_ema'] == 'True' else False
    #    predictor = EfficientNet_Predictor(weight_path, config_dict)
    #else:
    #    predictor = Predictor(weight_path, config_dict)

    predictor = Predictor(weight_path, config_dict)
    return predictor
    
        
        


#class Predictor(object):
#    """Classify images to predifined classes."""
#    
#    def __init__(self,
#                 checkpoint_path,
#                 config_dict):  
#        """Constructor.
#        
#        Args:
#            frozen_inference_graph_path: Path to frozen inference graph.
#            gpu_index: The GPU index to be used. Default None.
#        """
#        self._gpu_index = config_dict['GPU_OPTIONS']['GPU_DEVICES']
#        # Specify which gpu to be used.
#        if self._gpu_index is not None:
#            if not isinstance(self._gpu_index, str):
#                self._gpu_index = str(self._gpu_index)
#            os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_index
#        
#
#        print('Creating session and loading parameters')
#        with tf.Graph().as_default():
#            # setting not fully occupied memory, allocated on demand
#            config = tf.ConfigProto()
#            config.gpu_options.allow_growth = True
#            #config.gpu_options.per_process_gpu_memory_fraction = 0.50
#            sess = tf.Session(config = config)
#            with sess.as_default():
#                meta_path = checkpoint_path + '.meta'
#                saver = tf.train.import_meta_graph(meta_path)
#
#                #saver.restore(sess, './xxx/xxx.ckpt')
#                saver.restore(sess, checkpoint_path)
#
#                #op_list = sess.graph.get_operations()
#                #for op in op_list:
#                #    print(op.name)
#                #    print(op.values())
#
#                self._inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
#                self._is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
#                self._prediction = tf.get_default_graph().get_tensor_by_name('score_list:0')
#                #self._prediction = tf.get_default_graph().get_tensor_by_name('resnet_18/logits:0')
#
#                self._sess = sess
#        
#        
#    def predict(self, inputs):
#        """Predict prediction tensors from inputs tensor.
#        
#        Args:
#            preprocessed_inputs: A 4D float32 tensor with shape [batch_size, 
#                height, width, channels] representing a batch of images.
#            
#        Returns:
#            classes: A 1D integer tensor with shape [batch_size].
#        """
#        feed_dict = {self._inputs: inputs, self._is_training: False}
#        classes = self._sess.run(self._prediction, feed_dict=feed_dict)
#        return classes


class Predictor(object):
    """Classify images to predifined classes."""

    def __init__(self,
                 checkpoint_path,
                 config_dict):
        """Constructor.
        
        Args:
            frozen_inference_graph_path: Path to frozen inference graph.
            gpu_index: The GPU index to be used. Default None.
        """
        model_name = config_dict['MODEL']['NETWORK_NAME']
        image_size = config_dict['DATASET']['IMAGE_SIZE']
        import_str = import_model_by_networkname(model_name)
        scope = {}
        exec(import_str, scope)
        Model = scope['Model']


        ema_decay = config_dict['TRICKS']['MOVING_AVERAGE_DECAY']   # TODO: 有变化 'False' to 0.9
        enable_ema = False if ema_decay == -1 else True

        self._gpu_index = config_dict['GPU_OPTIONS']['GPU_DEVICES']
        # Specify which gpu to be used.
        if self._gpu_index is not None:
            if not isinstance(self._gpu_index, str):
                self._gpu_index = str(self._gpu_index)
            os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_index


        print('Creating session and loading parameters')
        with tf.Graph().as_default():
            # setting not fully occupied memory, allocated on demand
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            sess = tf.Session(config = config)
            with sess.as_default():
                inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='inputs')
                labels = tf.placeholder(tf.float32, shape=[None, 2], name='labels')
                #is_training = tf.placeholder(tf.bool, name='is_training')  # no use for efficientnet but need

                model = Model(config_dict)
                model.build_model(inputs, labels, is_training=False)

                self.restore_model(sess, checkpoint_path, enable_ema)

                self._inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
                #self._is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
                self._prediction = tf.get_default_graph().get_tensor_by_name('score_list:0')
                #self._prediction = tf.get_default_graph().get_tensor_by_name('resnet_18/logits:0')

                self._sess = sess



    def restore_model(self, sess, checkpoint, enable_ema=False):
        """Restore variables from checkpoint dir."""
        sess.run(tf.global_variables_initializer())
        #checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if enable_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.0)
            ema_vars = self.get_ema_vars()
            var_dict = ema.variables_to_restore(ema_vars)
            #ema_assign_op = ema.apply(ema_vars)
        else:
            var_dict = self.get_ema_vars()
            #ema_assign_op = None

        tf.train.get_or_create_global_step()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_dict)
        saver.restore(sess, checkpoint)


    def get_ema_vars(self):
        """Get all exponential moving average (ema) variables."""
        ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
        for v in tf.global_variables():
            # We maintain mva for batch norm moving mean and variance as well.
            if 'moving_mean' in v.name or 'moving_variance' in v.name:
                ema_vars.append(v)
        return list(set(ema_vars))


    def predict(self, inputs):
        """Predict prediction tensors from inputs tensor.
        
        Args:
            preprocessed_inputs: A 4D float32 tensor with shape [batch_size, 
                height, width, channels] representing a batch of images.
            
        Returns:
            classes: A 1D integer tensor with shape [batch_size].
        """
        #feed_dict = {self._inputs: inputs, self._is_training: False}
        feed_dict = {self._inputs: inputs}
        classes = self._sess.run(self._prediction, feed_dict=feed_dict)
        return classes

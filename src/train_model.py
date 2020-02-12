# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:35 2018

Train a CNN classification model.

@author: as 
"""
from __future__ import division
import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from data.data_provider import data_parallel_generator, get_mini_batch
from data.dataset_utils import load_dataset_mean_std_file, restore_normalization, restore_channel_normalization
from utils.config_utils import load_config_file, mkdir_if_nonexist, import_model_by_networkname
from utils.learning_rate_utils import get_learning_rate_scheduler
from utils.optimizer_utils import get_optimizer
from utils.noise_filter_utils import NoiseFilter


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string('config_path', '', 'path of the config file')
FLAGS = flags.FLAGS


 

def main(_):

    config_path = FLAGS.config_path
    config_dict = load_config_file(config_path)

    model_paras = config_dict['MODEL']
    gpu_paras = config_dict['GPU_OPTIONS']
    solver_paras = config_dict['SOLVER']
    train_paras = config_dict['TRAIN']
    dataset_paras = config_dict['DATASET']
    output_paras = config_dict['OUTPUT']
    trick_paras = config_dict['TRICKS']


    # gpu parameters 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_paras['GPU_DEVICES'])    # Specify which gpu to be used

    # model parameters
    network_name = model_paras['NETWORK_NAME']
    num_classes = model_paras['NUM_CLASSES']

    # train parameters
    model_ckpt_path = train_paras['PRETRAIN_WEIGHTS']                                               # Path to the pretrained model
    train_mode = train_paras['TRAIN_MODE']                 # train from scratch or finetune on imagenet pretrained model or continue training on previous model
    loss_name = train_paras['LOSS']
    monitor = train_paras['MONITOR']
    batch_size = train_paras['BATCH_SIZE']

    # solver parameters
    epoch_num = solver_paras['MAX_EPOCH_NUMS']
    lr_policy = config_dict['SOLVER']['LR_POLICY']
    analysis_loss = True if lr_policy == 'noise_labels_filter' else False
    
    # dataset parameters
    image_size = dataset_paras['IMAGE_SIZE']
    dataset_workers = dataset_paras['DATA_WORKERS']

    # output parameters
    output_base_dir = output_paras['OUTPUT_SAVE_DIR']
    experiment_base_dir = os.path.join(output_base_dir, output_paras['EXPERIMENT_NAME'])

    model_save_dir = os.path.join(experiment_base_dir, 'weights')
    log_save_dir = os.path.join(experiment_base_dir, 'log')
    tensorboard_summary_dir = os.path.join(log_save_dir, 'tensorboard_summary')
    mkdir_if_nonexist(tensorboard_summary_dir, raise_error=False)
    result_save_dir = os.path.join(experiment_base_dir, 'result')

    # get dataset mean std info
    mean_std_file = os.path.join(model_save_dir, 'dataset_mean_var.txt')
    dataset_rgb_mean, dataset_rgb_std = load_dataset_mean_std_file(mean_std_file)
    
    ckpt_max_save_num = output_paras['CKPT_MAX_SAVE_NUM']
    show_augment_data = output_paras['SHOW_AUGMENT_DATA']
    print_details_in_log = output_paras['PRINT_DETAILS_IN_LOG']
    if show_augment_data:
        augment_data_save_dir = os.path.join(result_save_dir, 'visual_data_augment')
        mkdir_if_nonexist(augment_data_save_dir, raise_error=False)

    # trick parameters
    moving_average_decay = trick_paras['MOVING_AVERAGE_DECAY']
    label_smooth_epsilon = trick_paras['LABEL_SMOOTH_EPSILON']
    
    # import model by network_name
    import_str = import_model_by_networkname(network_name)
    scope = {}
    exec(import_str, scope)
    Model = scope['Model']
    

    # create placeholders
    inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name='inputs')
    labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    # init network
    model = Model(config_dict)

    # build model
    model.build_model(inputs, labels, is_training=True)

    # get correlation op: predictions, accuracy, loss
    #logits = model.logits_op
    predictions = model.predict_op
    accuracy = model.accuracy_op
    loss = model.loss_op
    predict_loss = model.predicted_loss_op
    #regular_loss = model.regularized_loss_op
    
    # global train step counter
    global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
    
    # set optimizer
    optimizer = get_optimizer(config_dict, learning_rate)

    # get train step op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_step = optimizer.minimize(loss, global_step)
    # these three line can fix the low valid accuarcy bug when set is_training=False
    # this bug is cause by use of BN, see for more: https://blog.csdn.net/jiruiYang/article/details/77202674
    
    # set ExponentialMovingAverage
    use_moving_average = True if moving_average_decay > 0 else False
    if use_moving_average:
        ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay, num_updates=global_step)
        ema_vars = model.get_ema_vars()
        with tf.control_dependencies([train_step]):
            train_step = ema.apply(ema_vars)
    
    # init Saver to restore model
    if train_mode == 'train_from_scratch':
        pass
    else:
        # if train_mode is 'finetune_on_imagenet' or 'continue_training', get variables need restore
        variables_to_restore = model.get_vars_to_restore(train_mode) 
        saver_restore = tf.train.Saver(var_list=variables_to_restore) 

    # learning rate scheduler
    lr_scheduler = get_learning_rate_scheduler(config_dict)

    # init noise filter if use LR_POLICY: noise_labels_filter
    if analysis_loss:
        noise_filter = NoiseFilter(config_dict)

    ####################
    # start session
    ####################
    # init variables
    init = tf.global_variables_initializer()
    # config
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth=True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        sess.run(init)
        
        # init Saver to save model
        saver = tf.train.Saver(max_to_keep=ckpt_max_save_num)
        
        if train_mode == 'finetune_on_imagenet' or train_mode == 'continue_training':
            # Load the pretrained checkpoint file xxx.ckpt
            print("Load the pretrained checkpoint file {}".format(model_ckpt_path))
            saver_restore.restore(sess, model_ckpt_path)
        
        total_batch_num = 0
        total_best_monitor = 0 if monitor == 'accuracy' else 10000
        #cur_lr = init_learning_rate
        for epoch in range(epoch_num):
            print("start training epoch {0}...".format(epoch+1))
            sys.stdout.flush()
            epoch_start_time = time.time()

            # init data provider
            train_datagen = data_parallel_generator(config_dict, is_train=True)
            valid_datagen = data_parallel_generator(config_dict, is_train=False)
            train_step_num = train_datagen.__len__()
            valid_step_num = valid_datagen.__len__()
            get_train_batch = get_mini_batch(train_datagen, num_workers=dataset_workers)
            get_valid_batch = get_mini_batch(valid_datagen, num_workers=dataset_workers)

            ####################
            # training one epoch
            ####################
            # training batch by batch until one epoch finish
            batch_num = 0
            loss_sum = 0
            acc_sum = 0
            for batch_idx in range(train_step_num): 
                # get a new batch data
                try:
                    img_paths, images, groundtruth_lists = next(get_train_batch)
                    cur_lr = lr_scheduler.get_learning_rate(epoch, batch_idx, train_step_num)
                    print(cur_lr)

                    if show_augment_data == 1 and total_batch_num < 10:
                        for i in range(len(images)):
                            img = restore_channel_normalization(images[i], dataset_rgb_mean, dataset_rgb_std)
                            img = Image.fromarray(img.astype(np.uint8))
                            show_img_path = str(total_batch_num) + "_" + str(i) + ".jpg"
                            img.save(os.path.join(augment_data_save_dir, show_img_path))
                        
                    total_batch_num += 1
                    batch_num += 1
                except:
                    raise RuntimeError("generate data error, please check!")

                train_dict = {inputs: images, 
                                labels: groundtruth_lists,
                                learning_rate: cur_lr}
                #loss_, acc_, _ = sess.run([loss, accuracy, train_step], feed_dict=train_dict)
                loss_, predict_losses_, acc_, predictions_, _ = sess.run([loss, predict_loss, accuracy, predictions, train_step], feed_dict=train_dict)
                
                # print batch predict details in log
                if print_details_in_log == 1 and batch_num % 200 == 0:
                    for k in range(batch_size):
                        print(predictions_[k], groundtruth_lists[k])
                # record loss to find noise label
                if analysis_loss:
                    noise_filter.record_loss_tofind_noise_label(img_paths, predict_losses_)
                
                loss_sum += loss_
                average_loss = loss_sum / batch_num
                acc_sum += acc_
                average_acc = acc_sum / batch_num
                
                train_text = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(batch_num, average_loss, average_acc)
                print(train_text)
                sys.stdout.flush()
            epoch_end_time = time.time()
            print("epoch {0} training finished.".format(epoch + 1)) 
            print("total use time: {}s\n".format(int(epoch_end_time - epoch_start_time)))
            sys.stdout.flush()


            ####################
            # validation one epoch
            ####################
            print("start validation, please wait...")
            sys.stdout.flush()

            # valid batch by batch until validation dataset finish
            batch_num = 0
            acc_sum, acc_mean = 0, 0
            loss_sum, loss_mean = 0, 0
            predict_loss_sum, predict_loss_mean = 0, 0
            for batch_idx in range(valid_step_num): 
                # get a new batch data
                try:
                    img_paths, valid_images, valid_groundtruth_lists = next(get_valid_batch)
                    batch_num += 1
                except:
                    raise RuntimeError("generate data error, please check!")

                valid_dict = {inputs: valid_images, 
                              labels: valid_groundtruth_lists}
                
                valid_loss_, valid_predict_loss_, valid_acc_ = sess.run([loss, predict_loss, accuracy], feed_dict=valid_dict)
                acc_sum += valid_acc_
                loss_sum += valid_loss_
                predict_loss_sum += np.mean(valid_predict_loss_)

            # summary: compute mean accuracy
            acc_mean = acc_sum / batch_num
            loss_mean = loss_sum / batch_num
            predict_loss_mean = predict_loss_sum / batch_num
            print("validation finished. loss:{:.5f}, predict loss:{:.5f}, accuracy:{:.5f}".format(
                loss_mean, predict_loss_mean, acc_mean)) 
            sys.stdout.flush()
            # summary validation accuracy
            #valid_acc_summary.value.add(tag="valid_accuary", simple_value = acc_mean)
            #train_writer.add_summary(valid_acc_summary, epoch)
                     

            # check validation result
            find_better_model = False
            if monitor == 'accuracy':
                monitor_value = acc_mean 
                if monitor_value > total_best_monitor:
                    find_better_model = True
                    print("epoch {}: val_accuracy improved from {:.5f} to {:.5f}".format(epoch+1, total_best_monitor, monitor_value))
                    sys.stdout.flush()
                    total_best_monitor = monitor_value
            elif monitor == 'loss':
                monitor_value = predict_loss_mean
                if monitor_value < total_best_monitor:
                    find_better_model = True
                    print("epoch {}: val_predict_loss drop down from {:.5f} to {:.5f}".format(epoch+1, total_best_monitor, monitor_value))
                    sys.stdout.flush()
                    total_best_monitor = monitor_value
            else:
                raise RuntimeError("error use of parameter monitor: {}".format(monitor))

            # save checkpoint 
            if find_better_model:
                ckpt_name = network_name + "-epoch{0}.ckpt".format(epoch+1)
                model_save_path = os.path.join(model_save_dir, ckpt_name)
                saver.save(sess, model_save_path, global_step=global_step)
                print('save mode to {}'.format(model_save_path))
                sys.stdout.flush()
            else:
                print("epoch {}: val_acc did not improve from {}".format(epoch+1, total_best_monitor))
                sys.stdout.flush()

            # let gpu take a breath
            time.sleep(2) 
            print("\n\n")
            sys.stdout.flush()
    
        # analysis loss to find noise label
        if analysis_loss:
            noise_filter.summary_loss_info()
    print("training finish")


if __name__ == '__main__':
    tf.app.run()


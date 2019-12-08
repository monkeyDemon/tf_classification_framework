# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:35:27 2018

一个二分类问题的多功能性能测试程序demo

与predict_img_dir.py仅仅对指定目录下图像进行预测不同
recall_precision_test.py会对模型进行准召测试
绘制准召变化曲线，计算曲线下面积
从多个角度为模型比较和选择提供参考

@author: as
"""
from __future__ import print_function, division
import os
import cv2
import sys
import glob
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import predictor_pb
import predictor_ckpt
from utils.config_utils import load_config_file, mkdir_if_nonexist, import_model_by_networkname


flags = tf.app.flags
flags.DEFINE_string('config_path', '', 'path of the config file.')
flags.DEFINE_string('weight_mode', 'ckpt', 'type of the weight file')
flags.DEFINE_string('positive_img_dir', ' ',
                    'Path to positive images (directory).')
flags.DEFINE_string('negative_img_dir', ' ',
                    'Path to negative images (directory).')
flags.DEFINE_float('threshold', 0.8, 'threshold used to compute the recall and precision.')
FLAGS = flags.FLAGS


# TODO: 针对不同问题，预测函数需要进行简单修改，下面是两个示例
def detect(predictor, image_path, threshold):
    image_src = cv2.imread(image_path)
    # use different strategy for different size
    shape = image_src.shape
    if shape[0] <= 60 or shape[1] <= 60:
        is_hit = False  # ignore small image
        pred_label = [1, 0]
    else:
        image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
        pred_label = predictor.predict([image])[0]
        is_hit = True if pred_label[1] > threshold else False
    score = pred_label[1]
    return is_hit, score


def detect2(predictor, image_path, threshold):
    image_src = cv2.imread(image_path)
    # use different strategy for different size
    shape = image_src.shape
    width = shape[1]
    height = shape[0]

    if len(shape) == 2:
        image = cv2.cvtColor(image_src, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    
    # resize(maintain aspect ratio) 
    long_edge_size = 160
    if width > height:
        height = int(height * long_edge_size / width)
        width = long_edge_size
    else:
        width = int(width * long_edge_size / height)
        height = long_edge_size
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    # padding
    preprocess_image = np.zeros((long_edge_size, long_edge_size, 3))
    if width > height:
        st = int((long_edge_size - height)/2)
        ed = st + height
        preprocess_image[st:ed,:,:] = image
    else:
        st = int((long_edge_size - width)/2)
        ed = st + width
        preprocess_image[:,st:ed,:] = image

    # predict
    pred_label = predictor.predict([preprocess_image])[0]
    is_hit = True if pred_label[0] > threshold else False
    score = pred_label[0]
    return is_hit, score



if __name__ == "__main__":    

    weight_mode = FLAGS.weight_mode
    positive_img_dir = FLAGS.positive_img_dir
    negative_img_dir = FLAGS.negative_img_dir
    threshold = FLAGS.threshold

    config_path = FLAGS.config_path
    config_dict = load_config_file(config_path)

    output_paras = config_dict['OUTPUT']
    output_dir = os.path.join(output_paras['TEST_RESULT_SAVE_DIR'], output_paras['EXPERIMENT_NAME'])
    mkdir_if_nonexist(output_dir, raise_error=False)

    weight_path = ""
    model_save_dir = os.path.join(output_paras['MODEL_SAVE_DIR'], output_paras['EXPERIMENT_NAME'])
    if weight_mode == 'ckpt':
        weight_path = tf.train.latest_checkpoint(model_save_dir)
        #predictor = get_predictor(weight_path, config_dict)
        predictor = predictor_ckpt.Predictor(weight_path, config_dict)
    elif weight_mode == 'pb':
        print("wait to finish")
        predictor = predictor_pb.Predictor(weight_path, gpu_index=gpu_device)
    elif weight_mode == 'savemodel':
        print("wait to finish")
    elif weight_mode == 'trt':
        print("wait to finish")
    
    
    
    # compute recall & precision
    print("\n\n-------------------evaluate recall & precision--------------------")
    # save False Negative sample in recall_mis_save_path
    recall_mis_save_path = os.path.join(output_dir, 'recall_mis/')
    os.mkdir(recall_mis_save_path)
    # save False Positive sample in false_detect_save_path
    false_detect_save_path = os.path.join(output_dir, 'false_detect/')
    os.mkdir(false_detect_save_path)
    # save True Positive sample in true_detect_save_path
    true_detect_save_path = os.path.join(output_dir, 'true_detect/')
    os.mkdir(true_detect_save_path)
    
    TP_count = 0
    FP_count = 0
    FN_count = 0
    TN_count = 0
    
    # test positive images
    cnt = 0
    pos_evaluation_list = []
    for root, dirs, files in os.walk(positive_img_dir):
        for filename in files:
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
                sys.stdout.flush()
            image_path = os.path.join(root, filename)
            try:
                is_hit, score = detect2(predictor, image_path, threshold)
                pos_evaluation_list.append(score)
            except Exception,e:
                print("error occur: {}".format(repr(e)))
                continue
            if(is_hit == False):
                shutil.copy(image_path, recall_mis_save_path + str(score) + '_' + filename)
                FN_count += 1
            else:
                #shutil.copy(image_path, true_detect_save_path + str(score) + '_' + filename)
                TP_count += 1
    
    print('\n------------------------------------')
    # test negative images  
    cnt = 0
    neg_evaluation_list = []
    for root, dirs, files in os.walk(negative_img_dir):
        for filename in files:
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
                sys.stdout.flush()
            image_path = os.path.join(root, filename)
            try:
                is_hit, score = detect2(predictor, image_path, threshold)
                neg_evaluation_list.append(score)
            except Exception,e:
                print("error occur: {}".format(repr(e)))
                continue
            if(is_hit == True):
                shutil.copy(image_path, false_detect_save_path + str(score) + '_' + filename)
                FP_count += 1
            else:
                TN_count += 1    
    precision = TP_count / (TP_count + FP_count) * 100
    print('precision: %f' % precision)
    recall = TP_count / (TP_count + FN_count) * 100
    print('recall: %f' % recall)


    print("\n\n---------------不同置信度下准召计算----------------")
    interval = 0.001
    confidence_list = [c for c in np.arange(0, 1, interval)] # the confidence when model discrimination
    fig_save_name = "precision_recall_curve.jpg" 
    fig_save_path = os.path.join(output_dir, fig_save_name)

    save_conf_list = []
    precision_list = []
    recall_list = []
    for idx, conf in enumerate(confidence_list):
        threshold = conf
        TP_count = 0   
        FP_count = 0   
        FN_count = 0   
        TN_count = 0   
        for score in pos_evaluation_list:
            if score < threshold:
                FN_count += 1
            else:
                TP_count += 1
        for score in neg_evaluation_list:
            if score >= threshold:
                FP_count += 1
            else:
                TN_count += 1
        if TP_count > 0:
            precision = TP_count / (TP_count + FP_count) * 100
            recall = TP_count / (TP_count + FN_count) * 100
            save_conf_list.append(conf)
            precision_list.append(precision)
            recall_list.append(recall)
        else:
            # condifence过大或过小会出现TP=0
            # 这种情况会使绘制的图像看起来异常,这些数据没必要记录
            pass

    # save the visualization of the confidence's impact
    plt.plot(precision_list, recall_list, 'b')
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.savefig(fig_save_path)

    # save precision recall list
    pr_list_path = os.path.join(output_dir, 'pr_list.txt')
    np.savetxt(pr_list_path, (save_conf_list, precision_list, recall_list))

    # compute area
    area = 0
    stop_idx = len(precision_list) - 1 
    for idx, precision in enumerate(precision_list):
        if precision < 80:
            continue  # TODO: 可以通过设置筛选条件来观察某一阶段的曲线面积，从而帮助筛选模型
        if idx < stop_idx: 
            dx = precision_list[idx+1] - precision_list[idx] 
            recall = (recall_list[idx+1] + recall_list[idx])/2
            if dx > 0:
                area += recall * dx
    print("area of precision-recall curve: {}".format(area))



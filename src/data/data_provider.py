# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:27:37 2019

parallel data generator

@author: as
"""
from __future__ import division
import os
import sys
import PIL
import time
import random
import traceback
import numpy as np
import multiprocessing

import data.dataset_utils as dataset_utils
from data.data_augmentation import *



class data_parallel_generator():
    
    def __init__(self, config_dict, is_train=True):
        #self.polices = str2polices(polices_str)
        self.img_root_dir = config_dict['DATASET']['DATASET_ROOT_DIR']
        self.image_size= config_dict['DATASET']['IMAGE_SIZE']
        self.batch_size= config_dict['TRAIN']['BATCH_SIZE']
        self.train_data_proportion = config_dict['DATASET']['TRAIN_DATA_PROPORTION']
        self.dataset_random_seed = config_dict['DATASET']['DATASET_RANDOM_SEED']
        self.max_sample_per_class = config_dict['DATASET']['MAX_SAMPLE_PER_CLASS']
        self.augment_method = config_dict['DATASET']['AUGMENT_METHOD']
        self.is_train= is_train

        output_paras = config_dict['OUTPUT']
        experiment_base_dir = os.path.join(output_paras['OUTPUT_SAVE_DIR'], output_paras['EXPERIMENT_NAME'])
        model_save_dir = os.path.join(experiment_base_dir, 'weights')
        # get label file path
        label_file = os.path.join(model_save_dir, 'label.txt')
    
        # show the basic infomation of dataset
        label_idx = 0
        label_names = []
        for label_name in os.listdir(self.img_root_dir):
            print("label index: {}, label name: {}".format(label_idx, label_name)) 
            label_names.append(label_name)
            label_idx += 1
        self.classes= len(label_names)
        labels_to_class_names = dict(zip(range(len(label_names)), label_names))
        dataset_utils.write_label_file(labels_to_class_names, model_save_dir)

        # get dataset mean std info
        mean_std_file = os.path.join(model_save_dir, 'dataset_mean_var.txt')
        self.dataset_rgb_mean, self.dataset_rgb_std = dataset_utils.load_dataset_mean_std_file(mean_std_file)

        # set back color
        self.back_color = tuple([int(x) for x in self.dataset_rgb_mean])

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch 


    def on_epoch_end(self):
        if self.max_sample_per_class != -1:
            self.on_epoch_end_max_sample()
            return

        labels_dir_list = []
        for label_name in os.listdir(self.img_root_dir):
            labels_dir_list.append(os.path.join(self.img_root_dir, label_name))

        self.imgs_path_list = []
        self.labels = []
        for idx, label_dir in enumerate(labels_dir_list):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                label = np.zeros((self.classes))
                label[idx] = 1
                self.imgs_path_list.append(img_path)
                self.labels.append(label)

        # mix data from multiple labels(use the same seed to keep the order)
        np.random.seed(self.dataset_random_seed)
        np.random.shuffle(self.imgs_path_list)
        np.random.seed(self.dataset_random_seed)
        np.random.shuffle(self.labels)

        # 计算训练集和验证集的分割点
        total_nums = len(self.labels)
        split_idx = int(self.train_data_proportion * total_nums)
        if self.is_train == True:
            self.imgs_path_list = self.imgs_path_list[:split_idx]
            self.labels = self.labels[:split_idx]
            # shuffle
            seed = int(random.uniform(1,1000))
            np.random.seed(seed)
            np.random.shuffle(self.imgs_path_list)
            np.random.seed(seed)
            np.random.shuffle(self.labels)
        else:
            self.imgs_path_list = self.imgs_path_list[split_idx:]
            self.labels = self.labels[split_idx:]
            print(self.imgs_path_list[:5])  # TODO: check 顺序是否会变化

        self.num_of_examples = len(self.labels)
        self.steps_per_epoch = self.num_of_examples // self.batch_size - 1


    def on_epoch_end_max_sample(self):
        # TODO: wait to finish
        # has bug now! no use!
        raise RuntimeError("only use MAX_SAMPLE_PER_CLASS = -1 now, other value has bug")

        labels_dir_list = []
        for label_name in os.listdir(self.img_root_dir):
            labels_dir_list.append(os.path.join(self.img_root_dir, label_name))

        self.imgs_path_list = []
        self.labels = []
        for idx, label_dir in enumerate(labels_dir_list):
            cur_path_list = []
            cur_labels_list = []
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                label = np.zeros((self.classes))
                label[idx] = 1
                cur_path_list.append(img_path)
                cur_labels_list.append(label)

            # random sample $max_sample_per_class samples
            if len(cur_path_list) <= self.max_sample_per_class or self.max_sample_per_class == -1:
                rand_index_list = np.arange(len(cur_path_list))
            else:
                rand_index_list = np.random.choice(len(cur_path_list), self.max_sample_per_class, replace=True)
            for rand_idx in rand_index_list:
                img_path = cur_path_list[rand_idx]
                label = cur_labels_list[rand_idx]
                self.imgs_path_list.append(img_path)
                self.labels.append(label)
        
        # mix data from multiple labels(use the same seed to keep the order)
        np.random.seed(self.dataset_random_seed)
        np.random.shuffle(self.imgs_path_list)
        np.random.seed(self.dataset_random_seed)
        np.random.shuffle(self.labels)

        # 计算训练集和验证集的分割点
        total_nums = len(self.labels)
        split_idx = int(self.train_data_proportion * total_nums)
        if self.is_train == True:
            self.imgs_path_list = self.imgs_path_list[:split_idx]
            self.labels = self.labels[:split_idx]
            # shuffle
            seed = int(random.uniform(1,1000))
            np.random.seed(seed)
            np.random.shuffle(self.imgs_path_list)
            np.random.seed(seed)
            np.random.shuffle(self.labels)
        else:
            self.imgs_path_list = self.imgs_path_list[split_idx:]
            self.labels = self.labels[split_idx:]

        self.num_of_examples = len(self.labels)
        self.steps_per_epoch = self.num_of_examples // self.batch_size - 1



    def __getitem__(self, index):
        paths = []
        images = []
        labels = []

        # generate batch data(the code is compatible with format error image)
        success_num = 0
        cur_idx = index * self.batch_size
        while success_num < self.batch_size:
            img_path = self.imgs_path_list[cur_idx]
            label = self.labels[cur_idx]
            try:
                im = self._process_single_img(img_path)
            except Exception as e:
                print(e)
                traceback.print_exc()
                cur_idx += 1
                continue
            # add the success process img
            paths.append(img_path)
            images.append(im)
            labels.append(label)
            success_num += 1
            cur_idx += 1
     
        batch_paths = np.array(paths)
        batch_images = np.array(images)
        batch_labels = np.array(labels)
        return batch_paths, batch_images, batch_labels 


    def _process_single_img(self, img_path):
        img = PIL.Image.open(img_path, 'r')

        # make sure channel order is RGB
        img, _ = dataset_utils.process_image_channels(img)

        if self.is_train == True:
            if self.augment_method == 'none':
                img = self._fix_shape(img)
            elif self.augment_method == 'baseline':
                img = self._do_augment_baseline(img)
            elif self.augment_method == 'customize1': 
                img = self._do_augment_customize1(img)
            elif self.augment_method == 'customize2': 
                img = self._do_augment_customize2(img)
            else:
                raise RuntimeError("use unknown value of parameter AUGMENT_METHOD")
        else:
            img = self._fix_shape(img)
        
        # convert PIL 2 numpy
        img = np.array(img)

        # normalization
        img = dataset_utils.channel_normalization(img, self.dataset_rgb_mean, self.dataset_rgb_std)
        return img

    
    def _fix_shape(self, img):
        width, height = img.size
    
        # resize(maintain aspect ratio) 
        long_edge_size = self.image_size
        img = dataset_utils.maintain_aspect_ratio_resize(img, long_edge_size)
    
        # padding
        img_padd = dataset_utils.padding_image_square(img, self.back_color)
        return img_padd



    def _do_augment_baseline(self, image):
        long_edge_size = self.image_size
        # cutout
        if random.random() < 0.8:
            image = Cutout(image, 0.2, color=self.back_color)
        # fix the image shape to [size, size]
        image = self._fix_shape(image)
        # random crop with padding
        image = random_crop_with_padd(image, int(long_edge_size*0.05), long_edge_size, padd_value=self.back_color)
        # random flip 
        image = random_flip(image, left_right_probability=0.5, up_down_probability=0.05)
        return image


    def _do_augment_customize1(self, image):
        """ 自定义数据增强方法1
            与传统的crop方法比，多了一点多尺度的感觉
        """
        # random crop or padding
        if random.random() < 0.5:
            # crop
            image = random_crop(image, crop_probability=0.8, v=0.3)
        else:
            # padd
            image = random_padd(image, padd_probability=0.8, v=0.3, padd_value=self.back_color)
    
        # random flip 
        image = random_flip(image, left_right_probability=0.5, up_down_probability=0.05)
    
        # cutout
        if random.random() < 0.8:
            image = Cutout(image, 0.2, color=self.back_color)
    
        # fix the image shape to [size, size]
        image = self._fix_shape(image)
        return image


    def _do_augment_customize2(self, image):
        """ 自定义数据增强方法2
            进行比较强的数据增强
        """
        # -----color space transformation-----
    
        # 随机调整亮度/对比度
        image, change_flag = random_adjust_brightness(image, probability=0.7, max_intensity=0.4)
    
        # 随机调整色相  not implement 
        #image = random_adjust_hue(image, 0.3)
    
        # 图像饱和度
        image, change_flag = random_adjust_saturation(image, probability=0.3, max_intensity=0.5)
    
        # 随机添加均匀分布噪声  no implement 
        #image = random_uniform_noise(image, 0.3, img_shape)
    
        # 随机进行高斯滤波
        image, change_flag = random_gauss_filtering(image, probability=0.15)
    
        # 随机椒盐噪声
        #image, change_flag =  random_salt_pepper(image, probability=0.05, intensity=0.05)
    
        # -----position transformation-----

        # random crop or padding
        if random.random() < 0.5:
            image, change_flag = random_crop(image, crop_probability=0.8, max_intensity=0.3)
        else:
            image, change_flag = random_padd(image, padd_probability=0.8, max_intensity=0.3)
    
        # random flip 
        image, change_flag = random_flip(image, left_right_probability=0.5, up_down_probability=0.05)
    
        # 随机转置 not implement 
        #image = _transpose_image(image, 0.2)

        # 随机进行仿射变换 not implement
        # pass
    
        # 随机旋转
        image, change_flag = random_rotate(image, rotate_prob=0.1, rotate_angle_max=10) 
        
        # -----fix shape-----
        # fix the image shape to [size, size]
        image = self._fix_shape(image)
        return image



# -------------------------------------------------------------------------------------------
# 多进程并行喂数据
# -------------------------------------------------------------------------------------------


def generate_proc(batch_queue, generator, lock, share_dict):
    steps_per_epoch = share_dict['steps_per_epoch']
    while True:
        try:
            if batch_queue.full() == False:
                with lock:
                    batch_id = share_dict['batch_idx']
                    share_dict['batch_idx'] += 1
                if batch_id >= steps_per_epoch:
                    break    # finish, don't need to continue generating data

                # generate new batch data
                batch_paths, batch_images, batch_labels = generator(batch_id)
                generate_batch_data = {}
                generate_batch_data['batch_id'] =  batch_id
                generate_batch_data['batch_paths'] =  batch_paths
                generate_batch_data['batch_images'] =  batch_images
                generate_batch_data['batch_labels'] = batch_labels
                batch_queue.put(generate_batch_data)
            else:
                time.sleep(0.1)
        except Exception:
            print(traceback.format_exc())
            #raise RuntimeError("test")



def get_mini_batch(datagen, num_workers = 4, queue_max_size = 8):
    """获取batch数据
    使用方法 next(get_mini_batch)
    本质上是一个生产者消费者的模式。
    通过多进程并行的向一个公有的队列中写入各自生成的batch数据
    消费者调用本函数对生产的数据进行消耗(feed到CNN中进行训练)
    """
    steps_per_epoch = datagen.__len__()
    generator = datagen.__getitem__

    batch_queue = multiprocessing.Queue(queue_max_size)  # 生成的batch数据队列，进程间共享，共同向队列中写入各自生产的batch
    lock = multiprocessing.Lock()  # 进程锁，用来保护互斥资源batch_idx(防止对相同的图片进行数据增强)

    # 在主进程中创建一个字典用于共享一些关键变量: batch_idx(互斥资源), steps_per_epoch(read only)
    m = multiprocessing.Manager()
    share_dict = m.dict()
    share_dict['batch_idx'] = 0
    share_dict['steps_per_epoch'] = steps_per_epoch
    for _ in range(num_workers):
        proc = multiprocessing.Process(target=generate_proc, args=(batch_queue, generator, lock, share_dict))
        proc.start()

    try:
        has_feed = 0
        # 消耗数据直至一个完整的epoch结束(以steps_per_epoch标识)
        while has_feed < steps_per_epoch:
            while True:
                if not batch_queue.empty():
                    # 队列非空，取一个batch
                    batch_data = batch_queue.get()
                    batch_id = batch_data['batch_id']  # only use to debug
                    batch_paths = batch_data['batch_paths']
                    batch_images = batch_data['batch_images']
                    batch_labels = batch_data['batch_labels']
                    break
                else:
                    time.sleep(0.1)

            yield batch_paths, batch_images, batch_labels
    except Exception:
        print(traceback.format_exc())
    finally:
        print("epoch finish")


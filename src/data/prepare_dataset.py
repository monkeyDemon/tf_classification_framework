# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:12:43 2018

A batch verify image tool

After downloading a large amount of image data, usually we find that some 
images can not be open, which may be caused by network transmission errors. 
Therefore, before using these images, use this tool to verify the image data,
and move the unreadable image to the specified path.

@author: as
"""
import os
import sys
import cv2
import numpy as np
import shutil
import warnings
from PIL import Image
import tensorflow as tf
# raise the warning as an exception
warnings.filterwarnings('error') 

from utils.config_utils import load_config_file, mkdir_if_nonexist

flags = tf.app.flags
flags.DEFINE_string('config_path', '', 'path of the config file')
FLAGS = flags.FLAGS


# load config file
config_path = FLAGS.config_path
config_dict = load_config_file(config_path)
sys.stdout.flush()

reshape_size = config_dict['DATASET']['IMAGE_SIZE']
src_dir = config_dict['DATASET']['DATASET_ROOT_DIR']
use_channel_normalization = config_dict['DATASET']['USE_CHANNEL_NORMALIZATION'] 

output_paras = config_dict['OUTPUT']
experiment_base_dir = os.path.join(output_paras['OUTPUT_SAVE_DIR'], output_paras['EXPERIMENT_NAME'])
model_save_dir = os.path.join(experiment_base_dir, 'weights')
result_save_dir = os.path.join(experiment_base_dir, 'result')
error_save_dir = os.path.join(result_save_dir, 'error_format')

mkdir_if_nonexist(model_save_dir, raise_error=False)
mkdir_if_nonexist(result_save_dir, raise_error=False)
mkdir_if_nonexist(error_save_dir, raise_error=False)

# get datast mean_var file path
mean_var_file = os.path.join(model_save_dir, 'dataset_mean_var.txt')


cnt = 0
rm_cnt = 0
rgb_list = []
for root, dirs, files in os.walk(src_dir):
    for file_name in files:
        cnt += 1
        if cnt % 1000 == 0: 
            print(cnt)
            sys.stdout.flush()

        src_file = os.path.join(root, file_name)
        dst_file = os.path.join(error_save_dir, file_name)
        try:
            # check by PIL Image
            img_pil = Image.open(src_file)

            # check by opencv
            img_cv = cv2.imread(src_file)
            if type(img_cv) != np.ndarray:
                shutil.move(src_file, dst_file)
                rm_cnt += 1
                print('error when read by cv2!', file_name)
                sys.stdout.flush()
                continue

            # check channel number
            shape = img_cv.shape
            if len(shape) == 3:
                # this image is valid, reshape it
                height, width = shape[:2]
                if width > height:
                    height = int(height * reshape_size / width)
                    width = reshape_size
                else:
                    width = int(width * reshape_size / height)
                    height = reshape_size
                img_reshape = cv2.resize(img_cv, (width, height), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(src_file, img_reshape)

                # compute channel mean value
                r_mean = np.mean(img_reshape[:,:,2])
                g_mean = np.mean(img_reshape[:,:,1])
                b_mean = np.mean(img_reshape[:,:,0])
                rgb_list.append([r_mean, g_mean, b_mean])

            elif len(shape) == 2:
                # change channel num to 3 
                img_bgr = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR) 
                #img_bgr = cv2.merge((img_cv, img_cv, img_cv))
                cv2.imwrite(src_file, img_bgr)
                print("change {} from gray to rgb".format(file_name))
                sys.stdout.flush()

                # compute channel mean value
                mean_value = np.mean(img_cv)
                rgb_list.append([mean_value, mean_value, mean_value])
            else:
                shutil.move(src_file, dst_file)
                rm_cnt += 1
                print('channel number error!', file_name)
                sys.stdout.flush()

        except Warning:
            shutil.move(src_file, dst_file)
            rm_cnt += 1
            print('A warning raised!', file_name)
            sys.stdout.flush()
        except:
            shutil.move(src_file, dst_file)
            #os.remove(src_file)
            rm_cnt += 1
            print('Error occured!', file_name)
            sys.stdout.flush()


if use_channel_normalization == 0:
    mean_var_file = os.path.join(model_save_dir, 'dataset_mean_var.txt')
    with open(mean_var_file, 'w') as writer:
        writer.write("R_mean_std:" + str(128) + ':' + str(128) + '\n')
        writer.write("G_mean_std:" + str(128) + ':' + str(128) + '\n')
        writer.write("B_mean_std:" + str(128) + ':' + str(128) + '\n')
else:
    # compute dataset channel mean and std
    rgb_list = np.array(rgb_list)
    r_mean = np.mean(rgb_list[:,2])
    g_mean = np.mean(rgb_list[:,1])
    b_mean = np.mean(rgb_list[:,0])
    r_std = np.std(rgb_list[:,2])
    g_std = np.std(rgb_list[:,1])
    b_std = np.std(rgb_list[:,0])
    mean_var_file = os.path.join(model_save_dir, 'dataset_mean_var.txt')
    with open(mean_var_file, 'w') as writer:
        writer.write("R_mean_std:" + str(r_mean) + ':' + str(r_std) + '\n')
        writer.write("G_mean_std:" + str(g_mean) + ':' + str(g_std) + '\n')
        writer.write("B_mean_std:" + str(b_mean) + ':' + str(b_std) + '\n')


print('finish')
print("error number {}".format(rm_cnt))

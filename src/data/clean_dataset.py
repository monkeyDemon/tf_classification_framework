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
import numpy
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

src_dir = config_dict['DATASET']['DATASET_ROOT_DIR']
result_dir = config_dict['OUTPUT']['TEST_RESULT_SAVE_DIR']
error_save_dir = os.path.join(result_dir, 'error_format')

mkdir_if_nonexist(error_save_dir, raise_error=False)

rm_cnt = 0
for root, dirs, files in os.walk(src_dir):
    for file_name in files:
        src_file = os.path.join(root, file_name)
        dst_file = os.path.join(error_save_dir, file_name)
        try:
            # check by opencv
            img_cv = cv2.imread(src_file)
            if type(img_cv) != numpy.ndarray:
                print('type error!', file_name)
                sys.stdout.flush()
                shutil.move(src_file, dst_file)
                continue
            # check by PIL Image
            img = Image.open(src_file)

            # check channel number
            shape = img_cv.shape
            if len(shape) == 3:
                pass # this image is valid
            elif len(shape) == 2:
                # change channel num to 3 
                print("change {} from gray to rgb".format(file_name))
                sys.stdout.flush()
                img_rgb = cv2.merge((img_cv, img_cv, img_cv))
                cv2.imwrite(src_file, img_rgb)
            else:
                print('channel number error!', file_name)
                sys.stdout.flush()
                shutil.move(src_file, dst_file)
                continue
        except Warning:
            print('A warning raised!', file_name)
            sys.stdout.flush()
            shutil.move(src_file, dst_file)
        except:
            print('Error occured!', file_name)
            sys.stdout.flush()
            #shutil.move(src_file, dst_file)
            os.remove(src_file)
            rm_cnt += 1
print('finish')
print(rm_cnt)

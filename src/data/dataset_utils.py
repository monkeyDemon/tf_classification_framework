# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:38:13 2019

Contains utilities for downloading and converting datasets.

@author: cbb
"""
from __future__ import division

import os
import io
import sys
from PIL import Image
import tensorflow as tf


def load_dataset_mean_std_file(mean_std_file):
    print("load dataset mean & std file {}".format(mean_std_file))
    with open(mean_std_file, 'r') as f:
        r_mean, r_std = f.readline().rstrip().split(':')[1:] 
        g_mean, g_std = f.readline().rstrip().split(':')[1:]
        b_mean, b_std = f.readline().rstrip().split(':')[1:]
    rgb_mean = [float(r_mean), float(g_mean), float(b_mean)]
    rgb_std = [float(r_std), float(g_std), float(b_std)]
    print("dataset RGB mean: {} {} {}".format(rgb_mean[0], rgb_mean[1], rgb_mean[2]))
    print("dataset RGB std: {} {} {}".format(rgb_std[0], rgb_std[1], rgb_std[2]))
    return rgb_mean, rgb_std



def normalization(image):
    """ do normal normalization operation
    image = (image - 128) / 128
    image dtype: numpy.ndarray
    """
    image = (image - 128) / 128
    return image


def restore_normalization(image):
    """ restore the image from normal normalization to range [0,255]
    image dtype: numpy.ndarray float
    """
    image = image * 128
    image = image + 128
    return image


def channel_normalization(image, rgb_mean, rgb_std):
    """ do normalization operation by channel
    image = (image - [r, g, b]) / [r_std, g_std, b_std]
    image dtype: numpy.ndarray
    """
    image = (image - rgb_mean) / rgb_std
    return image


def restore_channel_normalization(image, rgb_mean, rgb_std):
    """ restore the image from channel normalization to range [0,255]
    image dtype: numpy.ndarray float
    """
    image = image * rgb_std
    image = image + rgb_mean
    return image


def process_image_channels(image):
    """ make sure channel order is RGB
    image dtype: PIL Image
    """
    process_flag = False
    if image.mode == 'RGBA':
        # process the 4 channels .png
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r,g,b))
        process_flag = True
    elif image.mode != 'RGB':
        # process the one channel image
        image = image.convert("RGB")
        process_flag = True
    return image, process_flag



def normal_resize(image, resize):
    """PIL Image resize
    resize: [width, height]
    """
    if resize is not None:
        image = image.resize((resize[0], resize[1]), Image.ANTIALIAS)  # PIL resize order is (width, height)
    return image


def maintain_aspect_ratio_resize(image, long_edge_size):
    """ resize and keep the original image's aspect ratio
    image: PIL Image
    long_edge_size: the long edge's size after resize
    """
    width, height = image.size
    if width > height:
        height = int(height * long_edge_size / width)
        width = long_edge_size
    else:
        width = int(width * long_edge_size / height)
        height = long_edge_size
    #image = image.resize((width, height), Image.ANTIALIAS)
    image = image.resize((width, height), Image.BILINEAR)
    return image


def padding_image_square(image, padd_value=(0,0,0)):
    """ padding the image to square 
    image size after padding is [long_edge_size, long_edge_size]
    image: PIL Image
    """
    width, height = image.size
    long_edge_size = width if width >= height else height

    img_padd = Image.new('RGB', (long_edge_size, long_edge_size), padd_value)
    if width > height:
        h_st = int((long_edge_size - height)/2)
        img_padd.paste(image, (0, h_st))
    else:
        w_st = int((long_edge_size - width)/2)
        img_padd.paste(image, (w_st, 0))
    return img_padd




def write_label_file(labels_to_class_names, dataset_dir,
                     filename='labels.txt'):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))




def read_label_file(label_file_path):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    label_file_path: The path of the file where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = label_file_path
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names


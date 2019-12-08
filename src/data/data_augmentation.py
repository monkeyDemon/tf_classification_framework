# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:27:37 2019

# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py

@author: as
"""
from __future__ import division
import os
import cv2
import time
import random
import numpy as np




# 随机调整亮度 
# 变暗/变亮的过程中实际上也调整了图片的对比度
# 增强效果比较自然，概率可以适当大一些
def random_adjust_brightness(image, probability=0.6, max_intensity=0.4):
    has_change = False
    if random.random() <= probability:
        has_change = True
    else:
        return image, has_change

    mean_value = np.mean(image)
    intensity =  1 + (random.random() * 2 - 1) * max_intensity 
    if intensity > 1:
        #get brighter
        if mean_value * intensity > 200:  # too bright, cancel
            return image, False
        image = image * intensity
        image = np.clip(image, a_max=255, a_min=0)
    else:
        # get darker
        if mean_value * intensity < 50:  # too dark, cancel
            return image, False
        image = image * intensity
        image = np.clip(image, a_max=255, a_min=0)
    return image, has_change



# 随机调整对比度
def random_adjust_contrast(image, probability=0.3, max_intensity=0.2):
    has_change = False
    if random.random() <= probability:
        has_change = True
    else:
        return image, has_change

    rows, cols, channels = image.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], image.dtype)

    # 随机对比度调整
    contrast_intensity = 1 + (random.random() * 2 - 1) * max_intensity
    brightness_intensity = 0  # 亮度调整设置为0
    image = cv2.addWeighted(image, contrast_intensity, blank, 1-contrast_intensity, brightness_intensity)
    return image, has_change


# 随机调整色相
def random_adjust_hue(image, probability=0.3):
    raise RuntimeError("wait to finish")


# 图像饱和度
# https://blog.csdn.net/Benja_K/article/details/95478100
def random_adjust_saturation(image, probability=0.3, max_intensity=0.5):
    src_dtype = image.dtype
    has_change = False
    if random.random() <= probability:
        has_change = True
    else:
        return image, has_change
    
    img = image * 1.0
    img_out = img

    increment = (random.random() * 2 - 1) * max_intensity
    img_min = img.min(axis=2)
    img_max = img.max(axis=2)
    
    #获取HSL空间的饱和度和亮度
    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value/2.0
    
    # s = L<0.5 ? s1 : s2
    mask_1 = L < 0.5
    s1 = delta/(value + 1e-10)
    s2 = delta/(2 - value + 1e-10)
    s = s1 * mask_1 + s2 * (1 - mask_1)
    
    # 增量大于0，饱和度指数增强
    if increment >= 0 :
        # alpha = increment+s > 1 ? alpha_1 : alpha_2
        temp = increment + s
        mask_2 = temp >  1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        
        alpha = 1/alpha -1 
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
    # 增量小于0，饱和度线性衰减
    else:
        alpha = increment
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
    
    
    # RGB颜色上下限处理(小于0取0，大于1取1)
    mask_3 = img_out  < 0 
    mask_4 = img_out  > 255
    img_out = img_out * (1-mask_3)
    img_out = img_out * (1-mask_4) + mask_4 * 255
    
    img_out = img_out.astype(src_dtype) 
    return img_out, has_change



# 随机添加均匀分布噪声  
def random_uniform_noise(image, probability=0.3):
    raise RuntimeError("wait to finish")


# 随机进行高斯滤波
def random_gauss_filtering(image, probability=0.3, kernel_size_list=[3,5,7], max_standard_deviation=1.5): 
    has_change = False
    if random.random() <= probability:
        has_change = True
    else:
        return image, has_change

    # random parameter
    size_idx = np.random.randint(0, len(kernel_size_list))
    kernel_size = kernel_size_list[size_idx]
    standard_deviation = random.random() * max_standard_deviation

    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), standard_deviation)
    return image, has_change


# 椒盐噪声
def random_salt_pepper(image, probability=0.05, intensity=0.05):  
    has_change = False
    if random.random() <= probability:
        has_change = True
    else:
        return image, has_change

    image = image.copy()
    height, width = image.shape[:2]
    SP_NoiseNum=int(intensity * width * height)
    for i in range(SP_NoiseNum): 
        rand_h = np.random.randint(0, height) 
        rand_w = np.random.randint(0, width) 
        rand_c = np.random.randint(0,3)
        if np.random.randint(0,1)==0: 
            image[rand_h, rand_w, rand_c] = 0 
        else: 
            image[rand_h, rand_w, rand_c] = 255 
    return image, has_change


    


# -----position transformation-----


# random crop
def random_crop(image, crop_probability=0.8, max_intensity=0.125):
    has_change = False
    if random.random() <= crop_probability:
        has_change = True
    else:
        return image, has_change

    # 随机变化幅度
    intensity = random.random() * max_intensity

    # 随机crop位置
    height, width = image.shape[:2]
    h_st = int(height * intensity * random.random())
    h_ed = height - 1 - int(height * intensity * random.random())
    w_st = int(width * intensity * random.random())
    w_ed = width - 1 - int(width * intensity * random.random())

    # check
    if h_st < h_ed and w_st < w_ed: 
        image = image[h_st:h_ed, w_st:w_ed, :]
        return image, has_change
    else:
        raise RuntimeError("may be has bug")
        return image, False



# random padding
def random_padd(image, padd_probability=0.5, max_intensity=0.125):
    has_change = False
    if random.random() <= padd_probability:
        has_change = True
    else:
        return image, has_change

    # 随机变化幅度
    intensity = 1 + random.random() * max_intensity

    shape = image.shape
    height, width = shape[:2]
    new_height = int(height * intensity)
    new_width = int(width * intensity)

    # padding
    back_image = np.zeros((new_height, new_width, 3))
    h_st = int((new_height - height)/2)
    h_ed = h_st + height
    w_st = int((new_width - width)/2)
    w_ed = w_st + width
    back_image[h_st:h_ed, w_st:w_ed, :] = image
    return back_image, has_change
    


# random flip 
def random_flip(image, left_right_probability=0.5, up_down_probability=0.3):
    has_change = False
    r = random.random()
    if r <= left_right_probability:
        image = cv2.flip(image, 1)#水平翻转
        has_change = True
    r = random.random()
    if r <= up_down_probability:
        image = cv2.flip(image, 0)#垂直翻转
        has_change = True
    return image, has_change
    

# 随机转置
def random_transpose_image(image, probability=0.2):
    raise RuntimeError("wait to finish")

# 随机旋转
def random_rotate(image, rotate_prob=0.3, rotate_angle_max=15):
    has_change = False
    if random.random() <= rotate_prob:
        has_change = True
    else:
        return image, has_change

    height, width = image.shape[:2]
    center = (width / 2, height / 2) #取图像的中点
    
    # 随机选择角度
    r = random.random() * 2 - 1
    angle = int(r * rotate_angle_max)
    scale = 1

    # rotate
    M = cv2.getRotationMatrix2D(center, angle, scale)#获得图像绕着某一点的旋转矩阵 
    rotated = cv2.warpAffine(image, M, (width, height))   #cv2.warpAffine()的第二个参数是变换矩阵,第三个参数是输出图像的大小
    return rotated, has_change


# 仿射变换
def random_affine_transform(image, probability):
    raise RuntimeError("wait to finish")
    # https://blog.csdn.net/thisiszdy/article/details/87028312

    #对图像进行变换（三点得到一个变换矩阵）
    # 我们知道三点确定一个平面，我们也可以通过确定三个点的关系来得到转换矩阵
    # 然后再通过warpAffine来进行变换
    img=cv2.imread("lena.png")
    cv2.imshow("original",img)
    
    rows,cols=img.shape[:2]
    
    point1=np.float32([[50,50],[300,50],[50,200]])
    point2=np.float32([[10,100],[300,50],[100,250]])
    
    M=cv2.getAffineTransform(point1,point2)
    dst=cv2.warpAffine(img,M,(cols,rows),borderValue=(255,255,255))



# --------------------------------------------------------------------------------------------


def _fix_shape(img):
    shape = img.shape
    width = shape[1]
    height = shape[0]

    # resize(maintain aspect ratio) 
    long_edge_size = 160
    if width > height:
        height = int(height * long_edge_size / width)
        width = long_edge_size
    else:
        width = int(width * long_edge_size / height)
        height = long_edge_size
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    # padding
    fix_img = np.zeros((long_edge_size, long_edge_size, 3))
    if width > height:
        st = int((long_edge_size - height)/2)
        ed = st + height
        fix_img[st:ed,:,:] = img
    else:
        st = int((long_edge_size - width)/2)
        ed = st + width
        fix_img[:,st:ed,:] = img
    return fix_img


def _do_augment(image):
    # -----color space transformation-----

    # 随机调整亮度/对比度
    #image, change_flag = random_adjust_brightness(image, probability=0.7, max_intensity=0.4)

    # 随机调整色相  no implement 
    #image = random_adjust_hue(image, 0.3)

    # 图像饱和度
    image, change_flag = random_adjust_saturation(image, probability=0.3, max_intensity=0.5)

    # 随机添加均匀分布噪声  no implement 
    #image = random_uniform_noise(image, 0.3, img_shape)

    # 随机进行高斯滤波
    #image, change_flag = random_gauss_filtering(image, probability=0.15)   

    # 随机椒盐噪声
    #image, change_flag =  random_salt_pepper(image, probability=0.05, intensity=0.05)

    # cutout no implement 
    #Regularizing Neural Networks by Penalizing Confident Output Distributions
    # https://openreview.net/forum?id=HkCjNI5ex


    # -----position transformation-----

    # random crop or padding
    #if random.random() < 0.5:
    #    image, change_flag = random_crop(image, crop_probability=0.8, max_intensity=0.3)
    #else:
    #    image, change_flag = random_padd(image, padd_probability=0.8, max_intensity=0.3)

    # random flip 
    #image, change_flag = random_flip(image, left_right_probability=0.5, up_down_probability=0.05)

    # 随机转置 no implement 
    #image = _transpose_image(image, 0.2)

    # 随机旋转
    #image, change_flag = random_rotate(image, rotate_prob=0.1, rotate_angle_max=10) 
    
    # fix the image shape to [size, size]
    image = _fix_shape(image)
    return image



if __name__ == "__main__":
    
    debug_img_dir = '/new_train_data/ansheng/porn_dataset/hx6/normal'
    debug_save_dir = 'visual_debug'
    debug_num = 1000

    cnt = 0 
    total_time = 0
    for img_name in os.listdir(debug_img_dir):
        if cnt == debug_num:
            break
        img_path = os.path.join(debug_img_dir, img_name)
        img = cv2.imread(img_path)
        
        st = time.time()
        img_aug = _do_augment(img)
        ed = time.time()

        img_aug = _fix_shape(img_aug)

        cnt += 1
        total_time += ed - st
        if cnt <= 200:
            save_path = os.path.join(debug_save_dir, img_name)
            save_path_aug = os.path.join(debug_save_dir, img_name[:-4] + '_aug.jpg')
            cv2.imwrite(save_path, img)
            cv2.imwrite(save_path_aug, img_aug)
    print("average use time: {} ms".format(total_time / cnt * 1000))


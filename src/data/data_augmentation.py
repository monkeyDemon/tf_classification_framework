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
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw



# -------------------------------------------------------------------------
# baseline data augmentation operator
# -------------------------------------------------------------------------

# random crop
def random_crop(image, crop_probability=0.8, v=0.125):
    if random.random() >= crop_probability:
        return image

    # 随机变化幅度
    intensity = random.random() * v

    # 随机crop位置
    width, height = image.size
    h_st = int(height * intensity * random.random())
    h_ed = height - 1 - int(height * intensity * random.random())
    new_h = h_ed - h_st + 1
    w_st = int(width * intensity * random.random())
    w_ed = width - 1 - int(width * intensity * random.random())
    new_w = w_ed - w_st + 1

    # check
    if new_w > 0 and new_h > 0: 
        crop_box = [w_st, h_st, new_w, new_h] 
        return image
    else:
        raise RuntimeError("may be has bug, check")
        return image

    image_cropped = image.crop(crop_box)
    return image_cropped


# random padding
def random_padd(image, padd_probability=0.8, v=0.125):
    if random.random() > padd_probability:
        return image

    # 随机变化幅度
    intensity = 1 + random.random() * v

    width, height = image.size
    new_width = int(width * intensity)
    new_height = int(height * intensity)

    # padding
    back_image = PIL.Image.new('RGB', (new_width, new_height), (0, 0, 0))
    h_st = int((new_height - height)/2)
    w_st = int((new_width - width)/2)
    
    #back_image.paste(image, (w_st, h_st, width, height))
    back_image.paste(image, (w_st, h_st))
    return back_image


# random flip
def random_flip(image, left_right_probability=0.5, up_down_probability=0.05):
    if random.random() <= left_right_probability:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    if random.random() <= up_down_probability:
        image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    return image


# -------------------------------------------------------------------------
# data augmentation operator in AutoAugment and Fast AutoAugment
# -------------------------------------------------------------------------
random_mirror = True

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    width, height = img.size
    max_edge = width if width > height else height
    v = v * max_edge
    #if width > height:
    #    v = v * height
    #else:
    #    v = v * width
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f




# ------------------------------------------------------------------------
# 以下是自定义的一些数据增强方法，可能有重复，需要整理

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
# 定义统一的调用接口

def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (Invert, 0, 1),  # 6
        (Equalize, 0, 1),  # 7
        (Solarize, 0, 256),  # 8
        (Posterize, 4, 8),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        # (SamplePairing(imgs), 0, 0.4),  # 15
    ]
    if for_autoaug:
        l += [
            (CutoutAbs, 0, 20),  # compatible with auto-augment
            (Posterize2, 0, 4),  # 9
            (TranslateXAbs, 0, 10),  # 9
            (TranslateYAbs, 0, 10),  # 9
        ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)



# --------------------------------------------------------------------------------------------
# 测试用函数


def _fix_shape(img):
    width, height = img.size

    # resize(maintain aspect ratio) 
    long_edge_size = 160
    if width > height:
        height = int(height * long_edge_size / width)
        width = long_edge_size
    else:
        width = int(width * long_edge_size / height)
        height = long_edge_size
    #img = img.resize((width, height), PIL.Image.ANTIALIAS)
    img = img.resize((width, height), PIL.Image.BILINEAR)

    # padding
    fix_img = PIL.Image.new('RGB', (long_edge_size, long_edge_size), (0, 0, 0))
    if width > height:
        h_st = int((long_edge_size - height)/2)
        fix_img.paste(img, (0, h_st))
    else:
        w_st = int((long_edge_size - width)/2)
        fix_img.paste(img, (w_st, 0))
    
    return fix_img


def _do_augment(image):

    # random crop or padding
    if random.random() < 0.5:
        # crop
        image = random_crop(image, crop_probability=0.8, v=0.3)
    else:
        # padd
        image = random_padd(image, padd_probability=0.8, v=0.3)

    # random flip 
    image = random_flip(image, left_right_probability=0.5, up_down_probability=0.05)

    # cutout
    if random.random() < 0.8:
        image = apply_augment(image, 'Cutout', 1)
    
    # fix the image shape to [size, size]
    image = _fix_shape(image)
    return image



if __name__ == "__main__":
    
    debug_img_dir = '/new_train_data/ansheng/porn_dataset/hx6/normal'
    debug_save_dir = 'visual_debug'
    debug_num = 200

    cnt = 0 
    total_time = 0
    for img_name in os.listdir(debug_img_dir):
        if cnt == debug_num:
            break
        img_path = os.path.join(debug_img_dir, img_name)
        img = PIL.Image.open(img_path, 'r')

        st = time.time()
        img_aug = _do_augment(img)
        ed = time.time()

        cnt += 1
        total_time += ed - st
        if cnt <= 200:
            save_path = os.path.join(debug_save_dir, img_name)
            save_path_aug = os.path.join(debug_save_dir, img_name[:-4] + '_aug.jpg')
            img.save(save_path)
            img_aug.save(save_path_aug)
    print("average use time: {} ms".format(total_time / cnt * 1000))

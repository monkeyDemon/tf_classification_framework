# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:05:58 2019

noise filter

@author: as
"""
import os
import shutil
import numpy as np
import pandas as pd
from utils.config_utils import mkdir_if_nonexist



class NoiseFilter(object):
    """noise filter"""

    def __init__(self, config_dict):
        """Constructor."""
        self.analysis_dict = {}
        self.config_dict = config_dict
        self.noise_doubt_save_num = config_dict['SOLVER']['POLICY_NOISE_FILTER']['DOUBT_SAVE_NUM']


    def record_loss_tofind_noise_label(self, img_paths, losses):
        for idx, img_path in enumerate(img_paths):
            loss = losses[idx] 
            if img_path not in self.analysis_dict:
                self.analysis_dict[img_path] = [loss]
            else:
                self.analysis_dict[img_path].append(loss)


    def summary_loss_info(self):
        print("summary_loss_info")
        info_list = []
        for img_path in self.analysis_dict.keys():
            losses = np.asarray(self.analysis_dict[img_path])
            mean = np.mean(losses)
            var = np.var(losses)
            info = [img_path, mean, var]
            info_list.append(info)
        df = pd.DataFrame(info_list, columns=['img_path', 'loss_mean', 'loss_var'])
        
        noise_save_dir = os.path.join(self.config_dict['OUTPUT']['TEST_RESULT_SAVE_DIR'], 'noise_label_analysis')
        mkdir_if_nonexist(noise_save_dir, raise_error=False)

        noise_label_analysis_path = os.path.join(noise_save_dir, 'noise_label_analysis.txt')
        with open(noise_label_analysis_path, 'w') as w:
            w.write("#loss_mean\n")   # 降序排列 loss mean
            df_sort_mean= df.sort_values(by='loss_mean' , ascending=False)
            for i in range(len(df_sort_mean)):
                row = df_sort_mean.iloc[i].values
                record = row[0] + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n'
                w.write(record)
    
            w.write("#loss_var\n")   # 降序排列 loss var
            df_sort_var = df.sort_values(by='loss_var' , ascending=False)
            for i in range(len(df_sort_var)):
                row = df_sort_var.iloc[i].values
                record = row[0] + '\t' + str(row[1]) + '\t' + str(row[2]) + '\n'
                w.write(record)
    
        # mkdir to save the doubt noise img
        doubt_img_save_dir = os.path.join(noise_save_dir, 'noise_doubt')
        mkdir_if_nonexist(doubt_img_save_dir, raise_error=False)
        img_root_dir = self.config_dict['DATASET']['DATASET_ROOT_DIR']
        for label_name in os.listdir(img_root_dir):
            label_dir = os.path.join(doubt_img_save_dir, label_name)
            mkdir_if_nonexist(label_dir, raise_error=False)

        # move the doubt img out of the train set
        doubt_set = set()
        with open(noise_label_analysis_path, 'r') as reader:
            cnt = 0
            for line in reader:
                if line.startswith('#loss_mean'):
                    cnt = 0
                    continue
                if line.startswith('#loss_var'):
                    cnt = 0
                    continue
                if cnt < self.noise_doubt_save_num:
                    cnt += 1
                    items = line.rstrip().split('\t')
                    img_path = items[0]
                    label_name = img_path.split('/')[-2]
                    img_name = img_path.split('/')[-1]
                    
                    if img_path in doubt_set:
                        continue
                    doubt_set.add(img_path)
                    label_dir = os.path.join(doubt_img_save_dir, label_name)
                    dst_path = os.path.join(label_dir, img_name)
                    shutil.move(img_path, dst_path)
    

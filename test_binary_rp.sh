#!/bin/bash

# test recall and precision of binary classification problem
echo "start predicting by src/test_binary_recall_precision.py ..."


config_path="configs/getting_started/tutorial_efficientnet-b0.yaml"
weight_mode='ckpt'   # ckpt
positive_img_dir='path of the positive test image directory'
negative_img_dir='path of the negative test image directory'
threshold=0.5


nohup python src/test_binary_recall_precision.py \
    --config_path $config_path \
    --weight_mode $weight_mode \
    --positive_img_dir $positive_img_dir \
    --negative_img_dir $negative_img_dir \
    --threshold $threshold \
    > log/test_log/test_binary_rp.out 2>&1 &

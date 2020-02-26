#!/bin/bash

# test recall and precision of binary classification problem
echo "start predicting by src/test_binary_recall_precision.py ..."


config_path="configs/getting_started/tutorial_efficientnet-b0.yaml"
weight_mode='ckpt'   # ckpt
positive_label_index=1 # index of the positive label, see labels.txt to check
positive_img_dir='path of the positive test image directory'
negative_img_dir='path of the negative test image directory'
threshold=0.5
ckpt_idx=-1


output_save_dir=$(cat ${config_path} | shyaml get-value OUTPUT.OUTPUT_SAVE_DIR)
experiment_name=$(cat ${config_path} | shyaml get-value OUTPUT.EXPERIMENT_NAME)
experiment_dir=${output_save_dir}'/'${experiment_name}
log_dir=${experiment_dir}'/log'
log_file=${log_dir}'/test_binary_rp.log'

if [ ! -d $output_save_dir ]; then
    mkdir $output_save_dir
fi
if [ ! -d $experiment_dir ]; then
    mkdir $experiment_dir
fi
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

echo "run src/test_binary_recall_precision.py"
nohup python src/test_binary_recall_precision.py \
    --config_path $config_path \
    --weight_mode $weight_mode \
    --positive_label_index $positive_label_index \
    --positive_img_dir $positive_img_dir \
    --negative_img_dir $negative_img_dir \
    --threshold $threshold \
    --ckpt_idx $ckpt_idx \
    > ${log_file} 2>&1 &
echo "test start..."

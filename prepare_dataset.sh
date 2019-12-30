#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./src


config_path="configs/getting_started/tutorial_efficientnet-b0.yaml"

output_save_dir=$(cat ${config_path} | shyaml get-value OUTPUT.OUTPUT_SAVE_DIR)
experiment_name=$(cat ${config_path} | shyaml get-value OUTPUT.EXPERIMENT_NAME)
experiment_dir=${output_save_dir}'/'${experiment_name}
log_dir=${experiment_dir}'/log'
log_file=${log_dir}'/prepare_dataset.log'

if [ ! -d $output_save_dir ]; then
    mkdir $output_save_dir
fi
if [ ! -d $experiment_dir ]; then
    mkdir $experiment_dir
else
    echo ${experiment_dir}' has exist, please check! Be careful!'
    exit 1
fi
if [ ! -d $log_dir ]; then
    mkdir $log_dir
fi

echo 'start cleaning dataset...'
nohup python src/data/prepare_dataset.py --config_path=${config_path} > ${log_file} 2>&1 &


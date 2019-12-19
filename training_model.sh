#!/bin/bash

# pip install shyaml

# TODO: you only need to modify the config path parameter here:
config_path="configs/getting_started/tutorial_efficientnet-b0.yaml"


output_save_dir=$(cat ${config_path} | shyaml get-value OUTPUT.OUTPUT_SAVE_DIR)
experiment_name=$(cat ${config_path} | shyaml get-value OUTPUT.EXPERIMENT_NAME)
experiment_dir=${output_save_dir}'/'${experiment_name}
log_dir=${experiment_dir}'/log'
log_file=${log_dir}'/train.log'

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

echo "run src/train_model.py"
nohup python src/train_model.py --config_path=${config_path} > ${log_file} 2>&1 &
echo "training start..."

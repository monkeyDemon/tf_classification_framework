#!/bin/bash


train_script="src/train.py"
config_path="configs/getting_started/tutorial_efficientnet-b0.yaml"
log_file="log/train_log/tutorial_efficientnet-b0.log"

echo "run ${train_script}"

nohup python ${train_script} \
    --config_path=${config_path} \
    > ${log_file} 2>&1 &

echo "training start..."

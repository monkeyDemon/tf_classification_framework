#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./src


config_path="configs/getting_started/tutorial_efficientnet-b0.yaml"

echo 'start cleaning dataset...'
nohup python src/data/clean_dataset.py \
	--config_path=${config_path} \
	> log/data_log/clean_dataset.out 2>&1 &

#!/bin/bash

# convert ckpt to savemodel format
# (savemodel format pb file is used for TF-Serving)

action='convert'  # print or convert
config_path="configs/getting_started/tutorial_efficientnet-b0.yaml"
ckpt_idx=-1

python src/convert_ckpt2savemodel.py \
    --action ${action} \
    --config_path=${config_path} \
    --ckpt_idx=${ckpt_idx}

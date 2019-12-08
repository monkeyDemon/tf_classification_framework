#!/bin/bash


# run GPU tensorflow/serving:

i=0  # choose GPU device
grpc_port=`expr 7000 + $i`
rest_port=`expr 8000 + $i`
docker run -e NVIDIA_VISIBLE_DEVICES=$i -e TF_FORCE_GPU_ALLOW_GROWTH=true --runtime=nvidia \
           -p $grpc_port:$grpc_port \
           -p $rest_port:$rest_port \
           --mount type=bind,source=/path of the/savemodel_base64,target=/models/model/modelname \
           --mount type=bind,source=/path of the/model.conf,target=/models/model.conf \
           --mount type=bind,source=/path of the/batching.conf,target=/models/batching.conf \
           -t tensorflow/serving:1.14.0-gpu \
           --port=$grpc_port --rest_api_port=$rest_port --model_config_file=/models/model.conf --tensorflow_intra_op_parallelism=28 --tensorflow_inter_op_parallelism=0 --rest_api_num_threads=28 --rest_api_timeout_in_ms=1000 --enable_batching=true --batching_parameters_file=/models/batching.conf --file_system_poll_wait_seconds=5 >> log &
echo "start docker instance on device $i, port:$grpc_port, rest_port:$rest_port"

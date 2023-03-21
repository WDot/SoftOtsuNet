#!/bin/bash

SEED=200
EXPNAME=test

docker run --mount type=bind,source=/home/,target=/home --rm --gpus 4 --env NCCL_DEBUG=INFO --env NCCL_SOCKET_IFNAME=eth0 --env NCCL_P2P_LEVEL=NODE --shm-size=8G --privileged=true -it equity:0.1 bash -c "cd [PATH-TO-CODE]; python3 main.py --batch_size=60 --test_batch_size=60 --lr=0.01 --epochs=250 --seed=$SEED --exp_name=${EXPNAME} --image_path=[PATH-TO-IMAGES] --label_path=[PATH-TO-LABELS]" 2>&1 | tee "${EXPNAME}.txt"
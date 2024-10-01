#!/bin/bash

docker buildx build --rm -t torch-test -f Dockerfile.test \
            --build-arg WANDB_SECRET=$(cat ~/.wandb_secret) \
            . &&
docker run -it --rm --gpus all --runtime=nvidia --privileged --ipc=host \
            -v ./src:/root/src \
            -v ./datasets:/root/datasets \
            -v ./history:/mnt/history \
            torch-test
            

#!/usr/bin/env bash

docker buildx build --rm -t torch-test -f Dockerfile.test \
            --build-arg WANDB_SECRET=$(cat ~/.wandb_secret) \
            . &&
docker run -it --rm --privileged --ipc=host \
            --device=nvidia.com/gpu=all \
            -v ./src:/root/src \
            -v ./datasets:/root/datasets \
            -v ./history:/mnt/history \
            torch-test
            

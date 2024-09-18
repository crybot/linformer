#!/bin/bash

docker buildx build --rm -t torch-test -f Dockerfile.test . &&
docker run -it --rm --gpus all --runtime=nvidia --privileged --ipc=host \
            -v ./src:/root/src \
            -v ./history:/mnt/history \
            torch-test
            

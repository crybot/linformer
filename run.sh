#!/bin/bash
# --shm-size 8G \

docker run -it \
  --rm \
  --gpus all \
  --runtime=nvidia \
  --privileged \
  --ipc=host \
  -p 8888:8888 \
  -v ./datasets:/root/HLT/datasets \
  -v ./artifacts:/root/HLT/artifacts \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  hlt-training \
  $@

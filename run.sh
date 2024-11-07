#!/usr/bin/env bash

docker run -it \
  --rm \
  --privileged \
  --device=nvidia.com/gpu=all \
  --ipc=host \
  -p 8888:8888 \
  -v ./datasets:/root/datasets \
  -v ./artifacts:/root/artifacts \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  hlt-training \
  $@

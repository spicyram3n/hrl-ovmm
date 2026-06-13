#!/bin/bash
# Build and run the SAM3 server. Run ON THE HOST.
# The facebook/sam3 checkpoint is gated: accept the license on Hugging Face
# and export HF_TOKEN before running.
#
#   export HF_TOKEN=hf_xxx
#   bash docker/sam3/run_sam3.sh
set -e

cd "$(dirname "$0")"
docker build -t hrl/sam3:latest .
docker run --gpus all -it --rm \
    --net=host \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    hrl/sam3:latest
# Server listens on http://localhost:5005/sam3/predict

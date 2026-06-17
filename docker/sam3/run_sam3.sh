#!/bin/bash
# Build and run the SAM3 server. Run ON THE HOST.
# The facebook/sam3 checkpoint is gated: accept the license on Hugging Face
# and put your token in docker/sam3/.env (gitignored) as:
#
#   HF_TOKEN=hf_xxx
#
# or export HF_TOKEN before running to override it.
set -e

cd "$(dirname "$0")"
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
docker build -t hrl/sam3:latest .
docker run --gpus all -it --rm \
    --net=host \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    hrl/sam3:latest
# Server listens on http://localhost:5005/sam3/predict

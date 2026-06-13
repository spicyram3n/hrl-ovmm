#!/bin/bash
# One-time setup: clone the Mask3D fork used by stretch-compose and
# download the ScanNet200 demo checkpoint.
# Run this ON THE HOST (not inside the devcontainer), from the repo root:
#   bash docker/mask3d/setup_mask3d.sh
set -e

MODELS_DIR="$(pwd)/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

if [ ! -d Mask3D ]; then
    git clone https://github.com/behretj/Mask3D.git
fi

mkdir -p Mask3D/checkpoints
cd Mask3D/checkpoints
if [ ! -f mask3d_scannet200_demo.ckpt ]; then
    wget "https://zenodo.org/records/10422707/files/mask3d_scannet200_demo.ckpt"
fi

echo "Mask3D ready at $MODELS_DIR/Mask3D"
echo "Now pull the docker image:  docker pull rupalsaxena/mask3d_docker:latest"

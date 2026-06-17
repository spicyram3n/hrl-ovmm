#!/bin/bash
# Pull and run the AnyGrasp (GraspNet) inference server. Run ON THE HOST.
#
# Same per-purpose-server pattern as docker/sam3 and docker/mask3d: runs in
# its own container, reached over REST via core/utils/rest_client.py using
# the "anygrasp" entry in configs/config.yaml.
#
# --mac-address is pinned because the AnyGrasp license used to build this
# image is tied to it - do not change it.
set -e

docker pull registry.gitlab.uni-bonn.de:5050/rpl/public_registry/graspnet:v1.0

docker run --net=bridge --mac-address=02:42:ac:11:00:02 \
    -p 5000:5000 --gpus all -it --rm \
    registry.gitlab.uni-bonn.de:5050/rpl/public_registry/graspnet:v1.0 \
    python3 app.py
# Server listens on http://localhost:5000/graspnet/predict
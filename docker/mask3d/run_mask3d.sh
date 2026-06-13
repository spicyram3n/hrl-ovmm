#!/bin/bash
# Run Mask3D segmentation on the data folder. Run ON THE HOST.
#
#   bash docker/mask3d/run_mask3d.sh [workspace] [mask3d_repo]
#
# workspace defaults to the host data folder (bind-mounted to /home/ws/data).
# mask3d.py looks for mesh.ply in the workspace, but the scanner here produces
# scene.ply, so we expose scene.ply under the name mesh.ply (symlink) for the
# run. Outputs (predictions.txt, pred_mask/, mesh_labeled.ply) land in the
# same folder.
#
# The image ENTRYPOINT is ["/bin/bash"], so the command is passed via -lc:
# -c so bash runs the string, -l so conda's (base) env activates.
set -e

WORKSPACE="${1:-/home/comrade/Desktop/data_hrl}"
MASK3D_REPO="${2:-$(pwd)/models/Mask3D}"

if [ ! -f "$WORKSPACE/scene.ply" ]; then
    echo "ERROR: $WORKSPACE/scene.ply not found."
    exit 1
fi
if [ ! -f "$MASK3D_REPO/checkpoints/mask3d_scannet200_demo.ckpt" ]; then
    echo "ERROR: checkpoint missing. Run docker/mask3d/setup_mask3d.sh first."
    exit 1
fi

# Expose scene.ply as mesh.ply (the name mask3d.py expects). Remove on exit
# only if we created it, so we never delete a real mesh.ply.
CLEANUP=""
if [ ! -e "$WORKSPACE/mesh.ply" ]; then
    ln -s scene.ply "$WORKSPACE/mesh.ply"
    CLEANUP="$WORKSPACE/mesh.ply"
fi
trap '[ -n "$CLEANUP" ] && rm -f "$CLEANUP"' EXIT

docker run --gpus all -it --rm \
    -v /home:/home \
    -w "$MASK3D_REPO" \
    rupalsaxena/mask3d_docker:latest \
    -lc "python3 mask3d.py --seed 42 --workspace '$WORKSPACE' --pcd"

echo "Done. Outputs in $WORKSPACE: predictions.txt, pred_mask/, mesh_labeled.ply"
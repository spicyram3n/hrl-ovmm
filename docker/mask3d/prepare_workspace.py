"""
prepare_workspace.py
--------------------
Turn one of the supervisor's raw .ply scans into a Mask3D workspace folder.

The Mask3D docker (rupalsaxena/mask3d_docker, running behretj/Mask3D's
mask3d.py) expects a workspace folder containing a file named `mesh.ply`.
It writes its outputs back into the same folder:

    <workspace>/
    ├── mesh.ply             <- input (this script puts it here)
    ├── mesh_labeled.ply     <- output: per-point instance colors
    ├── predictions.txt      <- output: "<mask_file> <scannet200_id> <conf>"
    └── pred_mask/           <- output: one 0/1 txt mask per instance
        ├── 000.txt
        └── ...

Run ON THE HOST or in the devcontainer (it only copies files):
    python docker/mask3d/prepare_workspace.py /path/to/supervisor_scan.ply \
        --data-dir /home/comrade/Desktop/data_hrl
"""

import argparse
import shutil
from pathlib import Path


def prepare(ply_path: Path, data_dir: Path, name: str | None = None) -> Path:
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    name = name or ply_path.stem
    workspace = data_dir / "prescans" / name
    workspace.mkdir(parents=True, exist_ok=True)

    target = workspace / "mesh.ply"
    if not target.exists():
        shutil.copy2(ply_path, target)
        print(f"Copied {ply_path} -> {target}")
    else:
        print(f"{target} already exists, leaving it untouched")

    print(f"Workspace ready: {workspace}")
    print("Next:  bash docker/mask3d/run_mask3d.sh " + str(workspace))
    return workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path, help="Path to the raw .ply scan")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/home/comrade/Desktop/data_hrl"),
                        help="Host data directory (bind-mounted to /home/ws/data)")
    parser.add_argument("--name", type=str, default=None,
                        help="Workspace folder name (default: ply filename)")
    args = parser.parse_args()
    prepare(args.ply, args.data_dir, args.name)

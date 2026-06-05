# HSR Scene Graph — Open-Vocabulary 3D Scene Understanding

Open-vocabulary 3D scene understanding for OVMM (Object-centric Vision and Mobile Manipulation) on the Toyota HSR-C robot. Built on OpenMask3D CLIP features to enable natural-language object queries from RGB-D scans.

---

## Scene Graph Output

![Scene Graph](images/sg.png)

*Output of `build_scene_graph.py` — movable objects (cups, bottles, etc.) connected to their nearest immovable anchor (table, shelf, etc.) with spatial edges. Visualized with Open3D.*

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Setup](#2-project-setup)
3. [Running the Dev Container](#3-running-the-dev-container)
4. [Running the OpenMask3D Container](#4-running-the-openmask3d-container)
5. [Running the Scene Graph](#5-running-the-scene-graph)

---

## 1. Prerequisites

On the **host machine**, install the following:

- [Docker](https://docs.docker.com/get-docker/) with NVIDIA Container Toolkit (`nvidia-docker2`)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- NVIDIA GPU with CUDA support
- ROS 2 Humble workspace built at `/home/comrade/hsr_ros2_ws`
- Scene data (RGB-D scan) at `/home/comrade/Desktop/data_hrl`

Allow Docker to access the X display (run once per session on the host):

```bash
xhost +local:docker
```

---

## 2. Project Setup

Clone this repository:

```bash
git clone <repo-url> /home/ws
cd /home/ws
```

The repo structure relevant to the pipeline:

```
/home/ws/
├── core/
│   └── perception/
│       ├── scene_graph/         # SceneGraph, ObjectNode, DrawerNode
│       └── segmentation/        # OpenMask3D query + server client
├── images/
│   └── sg.png                   # example scene graph output
├── .devcontainer/               # VS Code dev container config
│   ├── devcontainer.json
│   ├── Dockerfile
│   ├── initialize.sh
│   └── postCreate.sh
└── README.md
```

The following directories are **not tracked** in git and must exist on the host before opening the container:

| Host path | Mounted inside container at |
|---|---|
| your HSR ROS 2 workspace (built) | `/home/ws/ros2_ws` |
| your iPad scan data (from Rohit Menon) | `/home/ws/data` |

> **Before opening the container**, edit `.devcontainer/devcontainer.json` and update the `mounts` section to point to your local paths:
>
> ```json
> "mounts": [
>     "source=/home/comrade/hsr_ros2_ws,target=/home/ws/ros2_ws,type=bind,consistency=cached",
>     "source=/home/comrade/Desktop/data_hrl,target=/home/ws/data,type=bind,consistency=cached"
> ]
> ```
>
> Replace `source=/home/comrade/hsr_ros2_ws` with the absolute path to your own built HSR ROS 2 workspace, and `source=/home/comrade/Desktop/data_hrl` with the absolute path to the iPad scan data provided by Rohit Menon. The container will fail to start correctly if these paths do not exist on your machine.

---

## 3. Running the Dev Container

1. Open the `/home/ws` folder in VS Code.
2. When prompted, click **Reopen in Container** — or open the Command Palette (`Ctrl+Shift+P`) and run:
   ```
   Dev Containers: Reopen in Container
   ```
3. VS Code will build the Docker image (first time only, ~5–10 min) and start the container.

The container automatically:
- Sources `/opt/ros/humble/setup.bash` and `/home/ws/ros2_ws/install/setup.bash`
- Adds `/home/ws/source` to `PYTHONPATH`
- Installs Python dependencies: `open3d`, `openai`, `ikpy`, `graphviz`, `ultralytics`, `apriltag`, `CLIP`, `transformers`, `hydra-core`

To open a terminal inside the running container, use the VS Code integrated terminal.

---

## 4. Running the OpenMask3D Container

The OpenMask3D server runs **on the host** (outside the dev container) and exposes a REST API on port `5001`.

**In a host terminal** (not inside the dev container):

```bash
docker pull craiden/openmask:v1.0

docker run \
    -p 5001:5001 \
    --gpus all \
    -it \
    craiden/openmask:v1.0 python3 app.py
```

Wait until you see:
```
* Running on http://0.0.0.0:5001
```

The first run uploads the scene (~168 MB zip) and computes masks + CLIP features. This takes **4–6 hours on CPU** (GPU fallback is automatic on OOM). Results are cached to `data/openmask_features/` — subsequent runs load from cache in seconds and do not need the server.

**Then, from inside the dev container:**

```bash
cd /home/ws
python core/perception/segmentation/openmask_server.py
```

---

## 5. Running the Scene Graph

Once OpenMask3D features are cached, no server is needed.

**From inside the dev container:**

```bash
cd /home/ws
python core/perception/scene_graph/build_scene_graph.py
```

This will:
1. Load CLIP features and masks from `data/openmask_features/`
2. Run CLIP cosine similarity for each object label
3. Extract 3D point clusters via DBSCAN
4. Build spatial edges — each movable object connects to its nearest immovable anchor
5. Save `graph.json`, `scene.json`, and `objects/*.json` to `data/scene_graph/`
6. Open an Open3D visualization window (close it manually to proceed)

For detailed pipeline documentation, OpenMask3D configuration, troubleshooting, and the development roadmap, see [OPENMASK_README.md](OPENMASK_README.md).

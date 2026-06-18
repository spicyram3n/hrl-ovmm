"""
config.py
---------
Loads configs/config.yaml once and exposes its `servers:` section -
host/ip, port, and route for each per-purpose model server (sam3,
anygrasp, ...) that runs in its own docker container and is reached
over REST. Used by core/utils/rest_client.py.

Also exposes DATA_DIR/GRAPH_DIR, the on-disk scene graph location shared
by every script that builds or reads it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

# Scene graph data location, shared by the builders (build_scene_graph.py,
# build_scene_graph_gazebo.py), room_clustering.py and scene_query.py.
DATA_DIR = Path(os.environ.get("HRL_DATA_DIR", "/home/ws/data"))
GRAPH_DIR = DATA_DIR / "scene_graph"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_server_config(name: str) -> dict:
    """Return the `servers.<name>` entry (ip/port/route) from config.yaml."""
    servers = load_config().get("servers", {})
    if name not in servers:
        raise KeyError(f"No server '{name}' in {CONFIG_PATH} under `servers:`")
    return servers[name]

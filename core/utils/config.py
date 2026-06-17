"""
config.py
---------
Loads configs/config.yaml once and exposes its `servers:` section -
host/ip, port, and route for each per-purpose model server (sam3,
anygrasp, ...) that runs in its own docker container and is reached
over REST. Used by core/utils/rest_client.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


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

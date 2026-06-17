"""
rest_client.py
---------------
Shared base class for talking to the per-purpose model servers (SAM3,
AnyGrasp, ...) that each run in their own docker container and are
reached over REST (FastAPI + Uvicorn), as listed under `servers:` in
configs/config.yaml.

Handles the session, base URL, and error checking that every such client
needs. Subclasses add the request/response encoding specific to their
server (e.g. Sam3Client packs/unpacks masks+boxes+scores).

Usage:
    class Sam3Client(RestClient):
        def __init__(self, **kwargs):
            super().__init__(server="sam3", **kwargs)

        def detect(self, image_bgr, prompt):
            response = self.post(files=..., params=...)
            ...
"""

from __future__ import annotations

from typing import Optional

import requests

from core.utils.config import get_server_config


class RestClient:
    def __init__(self, server: Optional[str] = None, host: Optional[str] = None,
                 port: Optional[int] = None, route: Optional[str] = None,
                 timeout: int = 120):
        """
        server: name of an entry under `servers:` in configs/config.yaml
                (e.g. "sam3"). Supplies defaults for host/port/route, any
                of which can still be overridden explicitly.
        """
        cfg = get_server_config(server) if server else {}
        host = host or cfg.get("ip", "localhost")
        port = port or cfg.get("port")
        route = route if route is not None else cfg.get("route", "")
        if port is None:
            raise ValueError("port must be given explicitly or via `server` in config.yaml")

        self.base_url = f"http://{host}:{port}"
        self.url = f"{self.base_url}/{route.lstrip('/')}"
        self.timeout = timeout
        # Reuse one connection across calls - these clients are meant to be
        # called repeatedly (e.g. once per video frame).
        self.session = requests.Session()

    def post(self, **kwargs) -> requests.Response:
        """POST to `self.url`, raising on non-200 responses."""
        timeout = kwargs.pop("timeout", self.timeout)
        response = self.session.post(self.url, timeout=timeout, **kwargs)
        if response.status_code != 200:
            raise RuntimeError(f"{self.url} -> {response.status_code}: {response.text}")
        return response

    def healthy(self, path: str = "/healthz") -> bool:
        """Best-effort check of the server's health endpoint."""
        try:
            response = self.session.get(f"{self.base_url}{path}", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

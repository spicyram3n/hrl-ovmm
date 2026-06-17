"""
room_clustering.py  —  offline preprocessing step
--------------------------------------------------
Run this ONCE after building the scene graph to cluster furniture into rooms.
The output (rooms.json) is read at runtime by scene_query.predict_locations()
to add room-name context when the LLM predicts where to search for an object.

Pipeline:
    1. build scene graph  →  data/scene_graph/scene.json
    2. THIS SCRIPT        →  data/scene_graph/rooms.json   ← run once
    3. ros2 run fetcher search_and_fetch "<object>"        ← reads rooms.json

Input:  data/scene_graph/scene.json  (furniture with centroids)
Output: data/scene_graph/rooms.json
    {
        "rooms": [
            {"name": "living room", "furniture_ids": [2, 3, 4], "centroid": [x, y, z]}
        ]
    }

Usage:
    DEEPSEEK_API_KEY=<key> python -m core.llm_zone.room_clustering
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

DATA_DIR = Path(os.environ.get("HRL_DATA_DIR", "/home/ws/data"))
GRAPH_DIR = DATA_DIR / "scene_graph"

SYSTEM = (
    "You are a spatial reasoning agent. You receive a JSON list of furniture "
    "items, each with: id (int), label (str), centroid [x, y, z] in metres "
    "(map frame). Cluster them into rooms using both semantics (a stove and a "
    "refrigerator belong to a kitchen) and spatial proximity (items in one "
    "room are close together). Use common room names (kitchen, living room, "
    "bedroom, office, bathroom, hallway). Every furniture id must appear in "
    "exactly one room. Return ONLY valid JSON: "
    '{"rooms": [{"name": str, "furniture_ids": [int, ...]}]}'
)


def load_furniture(graph_dir: Path = GRAPH_DIR) -> list[dict]:
    with open(graph_dir / "scene.json") as f:
        scene = json.load(f)
    return [
        {"id": int(fid), "label": fdata["label"], "centroid": fdata["centroid"]}
        for fid, fdata in scene["furniture"].items()
    ]


def cluster_rooms(furniture: list[dict], model: str = "deepseek-chat") -> dict:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set the DEEPSEEK_API_KEY environment variable")
    client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({"furniture": furniture})},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    result = json.loads(raw)

    # attach a geometric centroid per room for navigation / visualization
    centroid_of = {f["id"]: np.array(f["centroid"]) for f in furniture}
    for room in result.get("rooms", []):
        ids = [i for i in room.get("furniture_ids", []) if i in centroid_of]
        room["centroid"] = (
            np.mean([centroid_of[i] for i in ids], axis=0).tolist() if ids else None
        )
    return result


def main() -> None:
    furniture = load_furniture()
    result = cluster_rooms(furniture)
    out = GRAPH_DIR / "rooms.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=4)
    print(json.dumps(result, indent=2))
    print(f"[rooms] saved to {out}")


if __name__ == "__main__":
    main()

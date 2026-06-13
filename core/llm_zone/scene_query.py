"""
scene_query.py
--------------
Open-vocabulary scene query using DeepSeek R1.
Loads the saved scene graph JSONs and asks the LLM which item best matches
the query — works for movable objects AND furniture/immovable items.

Usage:
    DEEPSEEK_API_KEY=<key> python core/llm_zone/scene_query.py "something to drink"
    DEEPSEEK_API_KEY=<key> python core/llm_zone/scene_query.py "a place to sit"
    DEEPSEEK_API_KEY=<key> python core/llm_zone/scene_query.py "book storage"

    Get a free API key at https://platform.deepseek.com

Returns JSON:
    {
        "present":      bool,
        "object_id":    int | null,
        "object_label": str | null,
        "type":         "movable" | "immovable" | null,
        "centroid":     [x, y, z] | null,   # map-frame, metres
        "reasoning":    str
    }

    centroid is the approach target for navigation:
      - immovable items  → their own centroid from scene.json
      - movable items    → centroid of the furniture they sit on
"""

import json
import os
import sys
from pathlib import Path
from openai import OpenAI

DATA_DIR = Path(os.environ.get("HRL_DATA_DIR", "/home/ws/data"))
GRAPH_DIR = DATA_DIR / "scene_graph"

SYSTEM = (
    "You are a scene graph reasoning agent. "
    "You receive a JSON list of 'items' — every object and piece of furniture in the scene. "
    "Each item has: id (int), label (str), type ('movable' or 'immovable'), "
    "and optionally on_furniture (str) for movable items. "
    "Given the user query, pick the single best matching item using open-vocabulary reasoning "
    "(e.g. 'a soft place to relax' → sofa; 'book storage' → shelf or cabinet; "
    "'something to drink' → bottle, cup, or mug). "
    "Return ONLY valid JSON: "
    '{"present": bool, "object_id": int|null, "object_label": str|null, "reasoning": str}'
)


def _load_raw() -> tuple[dict, dict]:
    with open(GRAPH_DIR / "graph.json") as f:
        graph = json.load(f)
    with open(GRAPH_DIR / "scene.json") as f:
        scene = json.load(f)
    return graph, scene


def _build_centroid_lookup(scene: dict, graph: dict) -> dict[int, list]:
    """Maps every item id → its map-frame centroid [x, y, z]."""
    lookup = {}
    furniture_centroids = {}

    for fid_str, fdata in scene["furniture"].items():
        fid = int(fid_str)
        lookup[fid] = fdata["centroid"]
        furniture_centroids[fid] = fdata["centroid"]

    # Movable objects inherit their furniture's centroid as the approach target
    for oid in graph["movable_ids"]:
        fid = graph["connections"].get(str(oid))
        if fid is not None and fid in furniture_centroids:
            lookup[oid] = furniture_centroids[fid]

    return lookup


def load_scene() -> tuple[list[dict], dict[int, list]]:
    """Returns (items_for_llm, centroid_lookup)."""
    graph, scene = _load_raw()

    furniture_map = {int(k): v["label"] for k, v in scene["furniture"].items()}
    centroid_lookup = _build_centroid_lookup(scene, graph)

    items = []

    for fid, label in furniture_map.items():
        items.append({"id": fid, "label": label, "type": "immovable"})

    for oid in graph["movable_ids"]:
        label = graph["node_labels"][graph["node_ids"].index(oid)]
        fid = graph["connections"].get(str(oid))
        items.append({
            "id": oid,
            "label": label,
            "type": "movable",
            "on_furniture": furniture_map.get(fid, "unknown"),
        })

    return items, centroid_lookup


def query(user_query: str) -> dict:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set the DEEPSEEK_API_KEY environment variable")

    client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=api_key)
    items, centroid_lookup = load_scene()

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps({"items": items}) + "\n\nQuery: " + user_query},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
        top_p=0.7,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    result = json.loads(raw)

    # Attach type and centroid from our lookup — LLM doesn't need to copy coordinates
    oid = result.get("object_id")
    if oid is not None:
        matched = next((it for it in items if it["id"] == oid), None)
        result["type"] = matched["type"] if matched else None
        result["centroid"] = centroid_lookup.get(oid)
    else:
        result["type"] = None
        result["centroid"] = None

    return result


PREDICT_SYSTEM = (
    "You are a household reasoning agent. You receive a JSON list of "
    "furniture items (id, label, room if known) present in a scene, and a "
    "target object that is NOT in the scene graph. Predict the top k most "
    "likely furniture locations where the object would be found, ordered by "
    "likelihood (e.g. a cup is most likely on the kitchen counter, then the "
    "dining table, then a shelf). Only use furniture ids from the list. "
    'Return ONLY valid JSON: {"predictions": [{"furniture_id": int, '
    '"furniture_label": str, "reasoning": str}]}'
)


def predict_locations(target: str, k: int = 3) -> list[dict]:
    """
    Scenario A: the target object is not in the scene graph.
    Ask the LLM for the top-k furniture locations to search, attach centroids.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set the DEEPSEEK_API_KEY environment variable")
    client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=api_key)

    items, centroid_lookup = load_scene()
    furniture = [it for it in items if it["type"] == "immovable"]

    # enrich with room names if rooms.json exists
    rooms_file = GRAPH_DIR / "rooms.json"
    if rooms_file.exists():
        with open(rooms_file) as f:
            rooms = json.load(f).get("rooms", [])
        room_of = {fid: r["name"] for r in rooms for fid in r.get("furniture_ids", [])}
        for it in furniture:
            it["room"] = room_of.get(it["id"], "unknown")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": PREDICT_SYSTEM},
            {
                "role": "user",
                "content": json.dumps({"furniture": furniture})
                + f"\n\nTarget object: {target}\nk: {k}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    predictions = json.loads(raw).get("predictions", [])[:k]

    for p in predictions:
        p["centroid"] = centroid_lookup.get(p.get("furniture_id"))
    return predictions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scene_query.py <query>")
        sys.exit(1)

    result = query(" ".join(sys.argv[1:]))
    print(json.dumps(result, indent=2))

    if not result.get("present"):
        print("\nObject not in scene graph. Predicting top-3 locations:")
        print(json.dumps(predict_locations(" ".join(sys.argv[1:]), k=3), indent=2))

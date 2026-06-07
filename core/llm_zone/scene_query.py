"""
scene_query.py
--------------
Open-vocabulary scene presence check using DeepSeek R1.
Loads the saved scene graph JSONs and asks the LLM whether an object
matching the query is present — no need to rebuild the scene.

Usage:
    DEEPSEEK_API_KEY=<key> python core/llm_zone/scene_query.py "something to drink"
    DEEPSEEK_API_KEY=<key> python core/llm_zone/scene_query.py "a place to sit"
"""

import json
import os
import sys
from pathlib import Path
from openai import OpenAI

GRAPH_DIR = Path("/home/ws/data/scene_graph")

SYSTEM = (
    "You are a scene graph reasoning agent. "
    "You receive a JSON scene graph (objects and the furniture they are on) and a user query. "
    "Decide whether any object in the graph matches the query — use open-vocabulary reasoning "
    "(e.g. 'something to drink' could match 'bottle', 'cup', 'mug'). "
    "Return ONLY valid JSON with this schema: "
    '{"present": bool, "object_id": int|null, "object_label": str|null, '
    '"on_furniture": str|null, "reasoning": str}'
)


def load_scene() -> dict:
    with open(GRAPH_DIR / "graph.json") as f:
        graph = json.load(f)
    with open(GRAPH_DIR / "scene.json") as f:
        scene = json.load(f)

    furniture = {int(k): v["label"] for k, v in scene["furniture"].items()}

    objects = []
    for oid in graph["movable_ids"]:
        label = graph["node_labels"][graph["node_ids"].index(oid)]
        fid = graph["connections"].get(str(oid))
        objects.append({
            "id": oid,
            "label": label,
            "on_furniture": furniture.get(fid, "unknown"),
        })

    return {"objects": objects, "furniture": list(furniture.values())}


def query(user_query: str) -> dict:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set the DEEPSEEK_API_KEY environment variable")

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    scene = load_scene()

    response = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(scene)},
                {"type": "text", "text": user_query},
            ]},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
        top_p=0.7,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content
    # strip chain-of-thought if present (response_format usually suppresses it, but just in case)
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    return json.loads(raw)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scene_query.py <query>")
        sys.exit(1)

    result = query(" ".join(sys.argv[1:]))
    print(json.dumps(result, indent=2))

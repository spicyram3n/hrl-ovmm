"""
scene_query.py
--------------
Open-vocabulary scene query using DeepSeek R1.
Loads the saved scene graph JSONs and asks the LLM which item best matches
the query — works for movable objects AND furniture/immovable items.

Usage:
    DEEPSEEK_API_KEY=<key> python core/reasoning/scene_query.py "something to drink"
    DEEPSEEK_API_KEY=<key> python core/reasoning/scene_query.py "a place to sit"
    DEEPSEEK_API_KEY=<key> python core/reasoning/scene_query.py "book storage"

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
import math
import sys

from core.reasoning._deepseek import get_client, strip_think
from core.utils.config import GRAPH_DIR

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
    if "movable_ids" not in graph:
        immovable_set = set(graph["immovable_ids"])
        graph["movable_ids"] = [i for i in graph["node_ids"] if i not in immovable_set]
    return graph, scene


# Furniture with exactly one flat resting surface, where "top of the
# bounding box" is a meaningful aim point. Deliberately excludes shelf/
# cabinet: those are tall multi-level units where an object can be on any
# internal level, so their bounding-box top is well above wherever the
# object actually is (e.g. a 1.8m shelf's top is the very top shelf, but an
# object two levels down is nowhere near it) - the bulk centroid (roughly
# the middle level) is the better single-point guess for those.
FLAT_SURFACE_LABELS = {"table", "desk", "coffee table", "trolley"}


def _furniture_top(fdata: dict) -> list:
    """Point on top of a furniture piece - where a resting object actually
    is, not the furniture's own bounding-box center. centroid[2] is the
    volumetric middle of the whole piece (mean of its points), so for
    anything taller than it is thin (a counter-height table) that center can
    be far below the surface: e.g. a 1.05m-tall table's centroid sits at
    z=0.525 - half the table's *height*, not its top. Only meaningful for
    FLAT_SURFACE_LABELS - see there for why shelf/cabinet are excluded."""
    cx, cy, cz = fdata["centroid"]
    height = fdata["dimensions"][2]
    return [cx, cy, cz + height / 2]


def _approach_target(fdata: dict) -> list:
    """Where to look for an object expected to be resting on this furniture:
    its top surface if it has exactly one (FLAT_SURFACE_LABELS), otherwise
    its bulk centroid - the safer single-point guess for multi-level
    storage like a shelf, where the object could be on any level."""
    if fdata.get("label") in FLAT_SURFACE_LABELS:
        return _furniture_top(fdata)
    return fdata["centroid"]


def _approach_radius(fdata: dict) -> float:
    """Conservative standoff radius for a furniture piece: half of its
    *shorter* horizontal dimension (width vs depth). The navigation layer
    adds its own clearance on top of this so it keeps a safe distance from
    the furniture's actual edge, not just its centroid point - a fixed
    standoff distance that works for a small side table would put the robot
    right up against a long counter's edge, and one safe for a long counter
    would be needlessly far from a small table. Using the *shorter*
    dimension (not the longer) deliberately assumes a roughly
    perpendicular approach to the long axis (you walk up to the side of a
    counter, not its end) - the longer dimension would force an
    unreachable standoff for any long counter/table/shelf."""
    w, d = fdata["dimensions"][0], fdata["dimensions"][1]
    return min(w, d) / 2


def _build_centroid_lookup(scene: dict, graph: dict) -> tuple[dict[int, list], dict[int, float]]:
    """Maps every item id → (its map-frame approach target [x, y, z], its
    approach radius - see _approach_radius).

    Furniture nodes map to their own centroid/radius (used when a query
    matches the furniture piece itself, e.g. "a place to sit" -> the couch).
    Movable objects map to their supporting furniture's approach
    target/radius instead - see _approach_target - since they rest on/in
    it, not at its center."""
    lookup = {}
    radius_lookup = {}
    furniture_data = {}

    for fid_str, fdata in scene["furniture"].items():
        fid = int(fid_str)
        lookup[fid] = fdata["centroid"]
        radius_lookup[fid] = _approach_radius(fdata)
        furniture_data[fid] = fdata

    # Movable objects inherit their furniture's approach target/radius
    for oid in graph["movable_ids"]:
        fid = graph["connections"].get(str(oid))
        if fid is not None and fid in furniture_data:
            lookup[oid] = _approach_target(furniture_data[fid])
            radius_lookup[oid] = _approach_radius(furniture_data[fid])

    return lookup, radius_lookup


def load_scene() -> tuple[list[dict], dict[int, list], dict[int, float]]:
    """Returns (items_for_llm, centroid_lookup, radius_lookup)."""
    graph, scene = _load_raw()

    furniture_map = {int(k): v["label"] for k, v in scene["furniture"].items()}
    centroid_lookup, radius_lookup = _build_centroid_lookup(scene, graph)

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

    return items, centroid_lookup, radius_lookup


def _nearest_by_label(items: list[dict], centroid_lookup: dict, matched: dict,
                      robot_position: list) -> dict:
    """Among every item sharing matched's exact label and type (e.g. two
    separate "pringles can" instances), return the one whose approach target
    is closest to robot_position (XY distance). The LLM that picked `matched`
    has no idea where the robot currently is, so when a label isn't unique,
    its choice between the duplicates is arbitrary - this replaces that
    arbitrary pick with a deterministic, distance-based one."""
    duplicates = [it for it in items
                 if it["label"] == matched["label"] and it["type"] == matched["type"]]
    rx, ry = robot_position[0], robot_position[1]

    def _dist(it: dict) -> float:
        c = centroid_lookup.get(it["id"])
        return math.inf if c is None else math.hypot(c[0] - rx, c[1] - ry)

    return min(duplicates, key=_dist)


def query(user_query: str, robot_position: list | None = None) -> dict:
    """robot_position: world-frame [x, y, (z)] of the robot, if known (see
    RobotAdapter.get_pose()). Used only to break ties when the query matches
    a label that isn't unique in the scene (e.g. two pringles cans) - without
    it, ties resolve to whichever the LLM happened to pick, with no
    preference for distance."""
    client = get_client()
    items, centroid_lookup, radius_lookup = load_scene()

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

    raw = strip_think(response.choices[0].message.content)
    result = json.loads(raw)

    # Attach type and centroid from our lookup — LLM doesn't need to copy coordinates
    oid = result.get("object_id")
    if oid is not None:
        matched = next((it for it in items if it["id"] == oid), None)
        result["type"] = matched["type"] if matched else None

        if matched and robot_position is not None:
            nearest = _nearest_by_label(items, centroid_lookup, matched, robot_position)
            if nearest["id"] != oid:
                result["reasoning"] = result.get("reasoning", "") + (
                    f" (multiple '{matched['label']}' instances exist; picked the one "
                    f"nearest to the robot's current position instead of the LLM's pick)")
                oid = nearest["id"]
                result["object_id"] = oid

        result["centroid"] = centroid_lookup.get(oid)
        result["approach_radius"] = radius_lookup.get(oid, 0.0)
    else:
        result["type"] = None
        result["centroid"] = None
        result["approach_radius"] = 0.0

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
    client = get_client()
    items, _, _ = load_scene()
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
    raw = strip_think(response.choices[0].message.content)
    predictions = json.loads(raw).get("predictions", [])[:k]

    # The predicted target is an object expected to be resting on/in this
    # furniture, not the furniture itself - use _approach_target/_approach_radius,
    # not the bulk furniture centroid, for the same reason movable objects do
    # in _build_centroid_lookup.
    _, scene = _load_raw()
    for p in predictions:
        fdata = scene["furniture"].get(str(p.get("furniture_id")))
        p["centroid"] = _approach_target(fdata) if fdata else None
        p["approach_radius"] = _approach_radius(fdata) if fdata else 0.0
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

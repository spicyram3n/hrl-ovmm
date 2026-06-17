# Scene Graphs Explained (Beginner's Guide)

This document explains, from scratch, what a "scene graph" is and how this
project builds one in `core/perception/scene_graph/`. It's written for
someone who is comfortable with basic Python but new to 3D geometry,
point clouds, and robotics perception. It uses the project's *own* code and
*real* output data (from `data/scene_graph/`) as examples throughout.

For the polished, demo-day-oriented technical write-up (with diagrams of the
Gazebo geometry pipeline), see
[`docs/report/scene_graph_report.pdf`](report/scene_graph_report.pdf). This
document is the "read this first" companion to that report.

---

## 1. What is a "scene graph"?

The term "scene graph" is overloaded:

- In **computer graphics / game engines**, a scene graph is a tree of
  transforms used for *rendering* (e.g. "the wheel is attached to the car,
  which is attached to the world").
- In **robotics / embodied AI** (this project's meaning), a scene graph is a
  **map of "what objects exist in the room, where they are, and how they
  relate to each other"**. It's the robot's internal model of its
  surroundings.

In this project, a scene graph is:

> A collection of **nodes** (one per object or piece of furniture), each
> with a 3D position, size, and orientation — plus a set of **connections**
> recording which movable objects are sitting on/near which furniture.

### Why does a robot need this?

Without a scene graph, the robot only has a raw camera image or a giant
point cloud — "a million 3D dots". A scene graph turns that into something
useful for *reasoning*:

- "Where is the apple?" → look up the node labeled `"apple"`, read its
  `centroid` (3D position).
- "What is the apple sitting on?" → follow its **connection** to find the
  furniture node it's linked to.
- "Which room is the kitchen?" → group furniture nodes by proximity +
  semantics (this is what `core/llm_zone/room_clustering.py` does, using an
  LLM, on top of the scene graph).

Everything downstream — navigation goals, "search and fetch" missions, LLM
queries — works on this compact graph, not on raw sensor data.

---

## 2. The two classes: `SceneGraph` and `ObjectNode`

The whole thing is built from two Python classes:

| Class | File | Represents |
|---|---|---|
| `ObjectNode` | `core/perception/scene_graph/graph_nodes.py` | **One object**: its points, position, size, orientation |
| `SceneGraph` | `core/perception/scene_graph/scene_graph.py` | **The whole map**: a collection of `ObjectNode`s plus the connections between them |

Think of `ObjectNode` as "one row in a spreadsheet describing one object",
and `SceneGraph` as "the spreadsheet, plus a notebook of relationships
between rows".

---

## 3. `ObjectNode`: anatomy of a single object

Every object in the scene — whether it's a real piece of furniture or a
banana — starts life as a **point cloud**: a list of 3D points `(x, y, z)`
sampled from that object's surface. `ObjectNode.__init__` takes that point
cloud and a few labels, and computes everything else automatically:

```python
# core/perception/scene_graph/graph_nodes.py
def __init__(self, object_id, color, sem_label, points, mesh_mask,
              confidence=None, movable=True):
    self.object_id = object_id      # unique integer id, e.g. 18
    self.color = color               # RGB color (mostly for visualization)
    self.sem_label = sem_label       # semantic label, e.g. "orange" or a class id
    self.centroid = np.mean(points, axis=0)   # <-- computed here
    self.points = points             # the (N, 3) point cloud
    self.mesh_mask = mesh_mask        # which points of the *full scene* belong to this object
    self.confidence = confidence      # how sure the detector was (1.0 = ground truth)
    self.movable = movable            # furniture = False, objects = True
    self.misplaced = False

    self.update_hull_tree()                       # <-- computed here
    self.compute_pose(self.points, self.centroid) # <-- computed here
    self.get_dimensions()                          # <-- computed here
```

Let's go through the four computed quantities one at a time, using the
**real orange** from this project's data
(`data/scene_graph/objects/18.json`) as a running example.

### 3.1 `centroid` — "where is it?"

```python
self.centroid = np.mean(points, axis=0)
```

This is just the **average position** of all the points — add up every
point's x, y, z and divide by how many points there are. It's the simplest
possible "center of the object".

For the orange:
```json
"centroid": [4.5749, 3.5577, 1.1127]
```
i.e. the orange is at x=4.57m, y=3.56m, z=1.11m in the world.

### 3.2 `pose` — "which way is it facing?" (PCA)

```python
def compute_pose(self, points, centroid):
    points_centered = points - centroid
    covariance_matrix = np.cov(points_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    R = eigenvectors
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    object_pose = np.eye(4)
    object_pose[:3, :3] = R
    object_pose[:3, 3] = centroid
    self.pose = object_pose
```

This is **Principal Component Analysis (PCA)**, and it sounds scarier than
it is. Here's the intuition:

1. **Center the points**: subtract the centroid, so the point cloud is now
   "centered on the origin".
2. **Compute the covariance matrix**: a 3×3 table that describes how spread
   out the points are along x, y, z — and whether that spread is
   *diagonal* (e.g. spread mostly along a tilted direction, not just
   along x or y).
3. **Find eigenvectors/eigenvalues**: the eigenvectors of the covariance
   matrix are the **three directions along which the point cloud is most
   spread out** (its "natural axes"); the eigenvalues say *how much*
   spread there is along each one.
4. **Sort by eigenvalue, descending**: the direction with the most spread
   becomes the object's "primary axis", etc.
5. Those three directions, as columns, form a 3×3 rotation matrix `R` — the
   object's **orientation**. Combined with the centroid (position), this
   becomes a 4×4 **pose matrix**:

```
pose = [ R[0,0]  R[0,1]  R[0,2]  centroid_x ]
       [ R[1,0]  R[1,1]  R[1,2]  centroid_y ]
       [ R[2,0]  R[2,1]  R[2,2]  centroid_z ]
       [   0       0       0         1      ]
```

This is the standard robotics convention for a **pose** (position +
orientation as one 4×4 matrix). The `if det(R) < 0: R[:, -1] *= -1` line
is a technicality: eigenvectors can come out as a "mirror image" rotation
(determinant -1); flipping the last axis turns it back into a proper
rotation (determinant +1).

> **Analogy**: imagine scattering the object's points on a table and
> drawing the tightest possible oval around them. PCA finds the long axis
> and short axis of that oval — for a banana, the long axis points along
> its length.

### 3.3 `dimensions` — "how big is it, and which way is 'up'?"

```python
def get_dimensions(self):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(self.points)
    obb = point_cloud.get_minimal_oriented_bounding_box()
    height_idx = np.argmax(np.abs(obb.R.T @ [0, 0, 1]))
    width_depth = sorted([i for i in range(3) if i != height_idx],
                          key=lambda i: obb.extent[i], reverse=True)
    order = width_depth + [height_idx]
    self.bb = obb
    self.dimensions = obb.extent[order]
```

An **oriented bounding box (OBB)** is the smallest box — allowed to be
*tilted* — that contains all the points. Think of it as shrink-wrapping the
object: a regular ("axis-aligned") bounding box can't rotate, so a tilted
banana would get a huge box; an *oriented* box rotates to fit snugly.
Open3D's `get_minimal_oriented_bounding_box()` computes this for us.

The box has an orientation `obb.R` (a rotation matrix, similar idea to
`pose`) and an `extent` (its size along each of its own 3 axes). The code
then figures out **which of those 3 axes points most "up"** (closest to
world `Z`) and calls that the **height**; the other two, sorted
largest-first, become **width** and **depth**.

For the orange:
```json
"dimensions": [0.066, 0.066, 0.1235]
```
→ width ≈ depth ≈ 6.6cm, height ≈ 12.35cm. (In Gazebo this orange model is
actually a sphere of radius 4.5cm/diameter 9cm — the extra height here
comes from how the point grid for this particular instance was sampled;
the key idea — "smallest box, height = axis closest to world up" — is what
matters for understanding the code.)

### 3.4 `hull_tree` — "fast nearest-point lookups on this object"

```python
def update_hull_tree(self) -> None:
    self.hull_tree = KDTree(self.points)
```

A **KD-tree** is a data structure for answering "what's the nearest point
to X?" quickly, without checking every single point one by one (it
organizes points into a tree by repeatedly splitting space in half). Here
it's built over *this object's own points*, so later we can ask "how close
is point P to *this specific object's surface*?" — used by
`SceneGraph.nearest_node()` (Section 4.4).

### 3.5 `transform()` — moving an object

```python
def transform(self, transformation: np.ndarray) -> None:
    ...
```

Given a translation `(3,)`, rotation `(3,3)`, or full transform `(4,4)`,
this updates `points`, `centroid`, and `pose` accordingly, then recomputes
`hull_tree` and `dimensions`. This is used if an object's position needs to
be updated after the graph was built (e.g. the world moved, or a node was
re-localized).

---

## 4. `SceneGraph`: the whole map

`SceneGraph` (in `scene_graph.py`) holds a dictionary of `ObjectNode`s plus
bookkeeping for connections and spatial queries.

```python
# core/perception/scene_graph/scene_graph.py
def __init__(self, label_mapping=None, min_confidence=0.0, k=2,
             immovable=None, pose=None):
    self.index = 0                   # next free node id
    self.nodes: dict[int, ObjectNode] = {}
    self.labels: dict[str, list[int]] = {}   # label -> [node ids with that label]
    self.outgoing: dict[int, int] = {}        # node id -> the ONE node it connects to
    self.ingoing: dict[int, list[int]] = {}   # node id -> [nodes that connect TO it]
    self.ids: list[int] = []
    self.label_mapping = label_mapping or {}  # raw label -> human-readable name
    self.immovable = immovable or []          # which human-readable labels are furniture
    self.tree: Optional[KDTree] = None        # KD-tree over ALL node centroids
```

### 4.1 `add_node()` — creating a node and deciding "furniture or object?"

```python
def add_node(self, color, sem_label, points, mesh_mask, confidence, movable=True):
    if self.label_mapping.get(sem_label, "ID not found") in self.immovable:
        self.nodes[self.index] = ObjectNode(self.index, np.array([0.5, 0.5, 0.5]),
                                             sem_label, points, mesh_mask, confidence, movable=False)
    else:
        self.nodes[self.index] = ObjectNode(self.index, color, sem_label,
                                             points, mesh_mask, confidence, movable=movable)
    self.labels.setdefault(sem_label, []).append(self.index)
    self.ids.append(self.index)
    self.index += 1
```

This:
1. Translates the raw `sem_label` (could be a string like `"table"` or a
   ScanNet200 integer class id) to a human-readable name via
   `label_mapping`.
2. If that name is in `self.immovable` (the furniture list), the new node
   is marked `movable=False` and given a fixed grey color — it's furniture,
   it won't get connected *to* anything, only connected *from*.
3. Otherwise it's a regular object (movable, unless the caller explicitly
   says `movable=False`).
4. Records the new id under its label, appends it to `self.ids`, and
   increments the id counter.

### 4.2 `update_connection()` — "what is this object on/near?"

```python
def update_connection(self, node: ObjectNode) -> None:
    min_index, min_dist = None, None
    if node.movable:
        for other in self.nodes.values():
            if not other.movable:
                dist = np.linalg.norm(node.centroid - other.centroid)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_index = other.object_id

    tmp = self.outgoing.get(node.object_id, None)
    if min_index is not None and tmp != min_index:
        if tmp is not None:
            self.ingoing[tmp].remove(node.object_id)
        self.outgoing[node.object_id] = min_index
        self.ingoing.setdefault(min_index, []).append(node.object_id)
```

For a **movable** node, this loops over *every other node*, computes the
straight-line distance between centroids, and finds the **closest
immovable (furniture) node**. That furniture's id becomes this node's one
**outgoing connection** — "I am on/near this piece of furniture". The
`ingoing` dict is just the reverse lookup ("which objects point at me?"),
kept in sync so it doesn't need to be recomputed from scratch.

Furniture nodes (`movable=False`) never get an outgoing connection — only
objects connect *to* furniture, never furniture to furniture or furniture
to objects.

### 4.3 The KD-tree over centroids (`self.tree`)

After all nodes are added and connected, the builder does:

```python
scene_graph.tree = KDTree(np.array([scene_graph.nodes[i].centroid for i in scene_graph.ids]))
```

This is a **second** KD-tree (don't confuse it with `ObjectNode.hull_tree`
from Section 3.4!) — built over the **centroids of all nodes in the whole
graph**. It powers:

- `query(point)` → id of the node whose centroid is closest to `point`.
- `get_centroid_distance(point)` → distance to that nearest centroid.
- `nearest_node(point)` (Section 4.4).

### 4.4 `nearest_node()` — finding the nearest *movable* object's surface

```python
def nearest_node(self, point):
    if point is None:
        return np.inf, None
    _, neighbor_indices = self.tree.query(point, k=4)
    neighbor_indices = [self.ids[n_idx] for n_idx in neighbor_indices
                         if self.nodes[self.ids[n_idx]].movable]
    if not neighbor_indices:
        return None, None
    nearest_neighbor = np.array([
        self.nodes[neighbor_idx].hull_tree.query(point, k=1)[0]
        for neighbor_idx in neighbor_indices
    ])
    return np.min(nearest_neighbor), neighbor_indices[np.argmin(nearest_neighbor)]
```

This combines **both** KD-trees: first, the graph-level `self.tree` finds
the 4 nodes whose *centroids* are nearest to `point` (cheap, coarse); then,
for whichever of those are movable, it uses *that node's own*
`hull_tree` (Section 3.4) to get the distance to its actual **surface**
(fine-grained). This two-step "coarse then fine" pattern is a common way to
make nearest-surface queries fast without checking every point in the
entire scene.

### 4.5 `remove_node()`, `color_with_ibm_palette()`, `visualize()`

- `remove_node(id)` deletes a node and patches up `outgoing`/`ingoing` for
  anything that pointed at it, then rebuilds `self.tree`.
- `color_with_ibm_palette()` assigns each *movable* node a random (but
  seeded, so reproducible) color from a fixed 10-color palette — purely
  cosmetic, for `visualize()`.
- `visualize()` opens an Open3D 3D window showing every node's points,
  optional centroids, optional connection lines, and optional text labels.
  Useful for sanity-checking that the graph looks like the real room.

---

## 5. Where do the points come from? Two builders

`ObjectNode`/`SceneGraph` don't care *where* the point clouds came from —
they just need `(points, sem_label, color, confidence, movable)` per
object. This project has **two** completely different ways of producing
that input, living in two separate scripts:

```
                Gazebo simulation                    Real RGB-D scan
            apartment.world.xacro                       scene.ply
                     |                                       |
                     v                                       v
        build_scene_graph_gazebo.py              docker/mask3d (Mask3D)
        + gazebo_geometry.py                              |
                     |                                     v
                     |                          mask3d_interface.py
                     |                                     |
                     v                                     v
                     +---------> build_scene_graph.py <----+
                                          |
                                          v
                              SceneGraph.add_node() for each object
                                          |
                                          v
                          update_connection() for every node
                          rebuild KD-tree, assign colors
                                          |
                                          v
                    save_all() -> graph.json, scene.json, objects/*.json
```

### 5.1 From Gazebo ground truth (`build_scene_graph_gazebo.py`)

This is the **simplest** path, and the best one to read first since there's
no machine-learning model involved — everything comes straight from files.

**Step 1 — read the world file.** `apartment.world.xacro` is a template
that expands (via the `xacro` CLI) into plain SDF/XML describing every
object placed in the simulated apartment, as `<include>` elements:

```xml
<include>
  <name>hsr_orange_02</name>
  <uri>model://hsr_orange</uri>
  <pose>4.6 3.9 1.05 0 0 0</pose>
</include>
```

`load_registry()` walks every `<include>` and reads its `name`, `uri`
(which model file to use), and `pose` (`x y z roll pitch yaw` — its
position and orientation in the world).

**Step 2 — translate names to labels.** `NAME_TO_LABEL` is a small
hand-written dictionary: `"hsr_orange_02" -> "orange"`. Anything *not* in
this dictionary (walls, doors, ...) is structural and gets skipped.

**Step 3 — figure out the object's size and shape from its own file.**
This is `gazebo_geometry.model_local_bbox(uri)`. Every Gazebo model has its
own `.sdf` file describing its physical shape for collision detection — a
box, cylinder, sphere, or 3D mesh. `model_local_bbox`:

1. Resolves `model://hsr_orange` to a folder on disk.
2. Reads that model's SDF, and for every shape inside it, computes the 8
   corners of that shape's bounding box (a sphere's bounding box is a cube
   of side `2*radius`, etc.).
3. Combines all those corners (handling each shape's own position/rotation
   within the model) into one overall box: `dims` (width, depth, height)
   and `center` (offset of the box's center from the model's origin).

This means **nothing about an object's size is typed in by a human** — if
the orange model in Gazebo is a sphere of radius 4.5cm, that 9cm diameter
is read directly from its SDF file. (One subtlety covered in the technical
report: mesh files in COLLADA `.dae` format declare their own unit of
measurement, which has to be applied — otherwise a mesh authored in
decimeters comes out 10× too big.)

**Step 4 — synthesize a point cloud.** `synthesize_points(pose, dims,
center)` builds a small 5×5×5 grid of points filling the box from Step 3,
then rotates and translates that grid using the `<include><pose>` from
Step 1 — turning "a box of this size, somewhere in the model's local frame"
into "125 actual 3D points in the world frame, exactly where the object
is".

**Step 5 — feed into `add_node`.** Those 125 points, plus the label from
Step 2 and whether that label is in `IMMOVABLE_LABELS_GZ`, become one call
to `scene_graph.add_node(...)`.

```python
# build_scene_graph_gazebo.py (simplified)
for entry in registry:
    points = synthesize_points(entry["pose"], entry["dims"], entry["center"])
    scene_graph.add_node(
        color=np.array([0.5, 0.5, 0.5]),
        sem_label=entry["label"],
        points=points,
        mesh_mask=None,
        confidence=1.0,
        movable=entry["label"] not in IMMOVABLE_LABELS_GZ,
    )
```

`confidence=1.0` because this is *ground truth* — there's no detector to be
uncertain about.

### 5.2 From a real scan (`build_scene_graph.py` + Mask3D)

This path is used for a real RGB-D scan instead of a simulation.

**Mask3D** is a neural network that takes a full 3D point cloud of a room
and outputs a set of **instances**: for each detected object, a binary mask
(which points belong to it), a class label (from the 200-category ScanNet200
label set), and a confidence score. `docker/mask3d/run_mask3d.sh` runs this
model and writes `predictions.txt` (one line per instance:
`<mask file> <class id> <confidence>`) plus one mask file per instance.

`mask3d_interface.py` loads these into a list of `Mask3DInstance` objects.

**The overlap problem.** Mask3D's instance masks can overlap — e.g. the
mask for "table" might include some points that are actually part of a cup
sitting on it. `build_scene_graph.py` resolves this so that **every point
belongs to exactly one object**: masks are processed from *largest to
smallest*, each overwriting the "owner" of its points — so a *smaller* mask
processed later "wins" its overlapping points back from a larger one. Net
effect: small objects on top of furniture get their points, and the
furniture keeps the rest.

```python
order = np.argsort([-m.sum() for m in masks])   # largest masks first
owner = np.full(n_pts, -1, dtype=np.int64)
for inst_idx in order:
    owner[masks[inst_idx]] = inst_idx
```

After that, for each instance, the points it owns (skipping tiny instances
with `< min_points`) become one `add_node()` call — same as the Gazebo
path, but with a real `confidence` value from the model and `sem_label` as
a ScanNet200 integer class id (translated to a name via `ID_TO_LABEL`).

From here on — `update_connection`, KD-tree, colors, `save_all` — **both
pipelines are identical**. That's the point of having a shared
`SceneGraph`/`ObjectNode`: it doesn't matter whether the points came from a
simulator or a real camera.

---

## 6. Worked example: the orange (node 18)

Let's trace one real object through the whole pipeline using actual numbers
from `data/scene_graph/`.

1. **World file**: `apartment.world.xacro` has
   `<include><name>hsr_orange_02</name><uri>model://hsr_orange</uri>
   <pose>... </pose></include>`.
2. **Label lookup**: `NAME_TO_LABEL["hsr_orange_02"] = "orange"`. `"orange"`
   is not in `IMMOVABLE_LABELS_GZ`, so this will be a **movable** object.
3. **Geometry**: `model_local_bbox("model://hsr_orange")` reads the
   orange's SDF (a sphere collision shape) and returns its bounding box
   `dims`/`center`.
4. **Point synthesis**: `synthesize_points(pose, dims, center)` produces
   125 points around the world position of the orange, rotated by the
   include's pose.
5. **`add_node`**: creates `ObjectNode(18, ..., sem_label="orange", ...,
   movable=True)`. Inside `__init__`:
   - `centroid = mean(points) = [4.5749, 3.5577, 1.1127]`
   - `pose` = PCA orientation + this centroid, as a 4×4 matrix
   - `dimensions = [0.066, 0.066, 0.1235]` (from the minimal OBB)
   - `hull_tree = KDTree(points)`
6. **`update_connection`**: loops over all *other* nodes, computes
   `|orange.centroid - other.centroid|` for every **immovable** node, and
   finds the minimum. The table (node 14, `centroid = [4.60, 4.00, 0.52]`)
   is the closest furniture — distance ≈ `sqrt(0.025² + 0.44² + 0.59²)` ≈
   `0.74m`, much closer than e.g. the shelves at `y ≈ 13`. So:
   ```python
   self.outgoing[18] = 14   # "orange is on the table"
   self.ingoing[14] = [18, 20, ...]   # table has things on it
   ```
7. **Output**: this connection is written into `graph.json`'s
   `"connections"` map as `"18": 14`, and the orange's own data into
   `objects/18.json`:

```json
{
    "id": 18,
    "label": "orange",
    "centroid": [4.5749, 3.5577, 1.1127],
    "dimensions": [0.066, 0.066, 0.1235],
    "pose": [[...], [...], [...], [0,0,0,1]],
    "drawer": -1,
    "confidence": 1.0
}
```

---

## 7. The output files

`SceneGraph.save_all(graph_dir)` writes three things to
`data/scene_graph/`:

| File | Contents | Who reads it |
|---|---|---|
| `graph.json` | `node_ids`, `node_labels`, `connections` (the `outgoing` map), `immovable_ids`/`immovable_labels` | `core/llm_zone/scene_query.py` (to know what's "on" what) |
| `scene.json` | `{furniture_id: {label, centroid, dimensions}}` — **immovable nodes only** | `core/llm_zone/room_clustering.py` (clusters furniture into rooms by label + centroid) |
| `objects/{id}.json` | One file per **movable** node: `id, label, centroid, dimensions, pose, drawer, confidence` | `core/llm_zone/scene_query.py`, `core/missions/search_and_fetch.py` |

The two LLM modules (`room_clustering.py`, `scene_query.py`) never see raw
point clouds or geometry math — only this compact JSON. That's the whole
point of building the scene graph: it's the **interface** between "3D
geometry" and "language-model reasoning about the room".

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Point cloud** | A set of 3D points `(x, y, z)`, usually representing a surface |
| **Centroid** | The average position of a set of points — its "center of mass" |
| **Covariance matrix** | A 3×3 table describing how spread out points are along/between x, y, z |
| **Eigenvector / eigenvalue** | The "natural axes" of a covariance matrix, and how much spread is along each one — used by PCA |
| **PCA (Principal Component Analysis)** | Finding an object's natural orientation from the spread of its points |
| **Pose** | Position + orientation together, usually as a 4×4 matrix |
| **(Oriented) Bounding Box** | The smallest (possibly rotated) box containing a set of points |
| **KD-tree** | A data structure for fast "nearest point" queries |
| **SDF** | Simulation Description Format — Gazebo's XML format for describing a model's shape, used for physics/collision |
| **Xacro** | An XML macro language that expands into SDF/URDF (used for `apartment.world.xacro`) |
| **Semantic label** | A human-meaningful category name for an object, e.g. `"chair"`, `"orange"` |
| **Confidence** | How sure a detector is that an instance is correctly segmented/classified (1.0 = ground truth, no detector) |
| **Movable / immovable** | Whether an object can be picked up (movable, e.g. an apple) or is furniture (immovable, e.g. a table) |
| **Connection (`outgoing`/`ingoing`)** | A one-way edge from a movable object to the nearest immovable furniture node |

---

## 9. Suggested next steps (learning by doing)

1. **Run it and look at the output.**
   ```bash
   python -m core.perception.scene_graph.build_scene_graph_gazebo --visualize
   ```
   Compare the printed table and the Open3D window to `data/scene_graph/graph.json`.

2. **Pick one object and trace it**, the way Section 6 did for the orange —
   e.g. the banana (id 22) or a chair (id 8/9/10/11). Find its
   `<include>` in `apartment.world.xacro`, its entry in `NAME_TO_LABEL`,
   and its JSON output.

3. **Add print statements.** In `ObjectNode.__init__`, temporarily print
   `self.centroid`, `self.pose[:3,:3]`, `self.dimensions` for one object and
   re-run — watching real numbers come out of the PCA/OBB code is the
   fastest way to build intuition for what those functions actually do.

4. **Read in this order**:
   - `core/perception/scene_graph/graph_nodes.py` (small, self-contained — Section 3)
   - `core/perception/scene_graph/scene_graph.py` (Section 4)
   - `core/perception/scene_graph/build_scene_graph_gazebo.py` (Section 5.1 — no ML involved)
   - `core/perception/scene_graph/gazebo_geometry.py` (the geometry details, Section 5.1 step 3)
   - `core/perception/segmentation/mask3d_interface.py` and `build_scene_graph.py` (Section 5.2)

5. **Try a small modification**: add a new `<include>` to a copy of the
   world file with an object you choose, add it to `NAME_TO_LABEL`, and
   re-run the builder — see your object appear in `graph.json`.

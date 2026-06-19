# Session changelog — Gazebo/Nav2 frame fix, fetcher pipeline, AnyGrasp swap, cleanup

This documents everything changed across this session, in the order it happened, in plain language. Scope note: every edit below is inside `fetcher`, `grasper`, `core/`, `scripts/`, or `docker/` — the official HSR/TMC ROS packages under `ros2_ws/src/` were left alone except for two explicitly-requested, one-off changes (called out below).

## 1. Removed unused doors from the apartment world

**Files:** `ros2_ws/src/tmc_gazebo/tmc_gazebo_worlds/worlds/apartment.world`, `apartment_fast.world` *(official package — explicitly requested)*

Removed 5 door models (`door_x03y09`, `door_x05y09a`, `door_x08y09a`, `door_x10y11`, `door_x08y13`) and their matching door-stopper models from both Gazebo world files. Documented in `tmc_gazebo_worlds/README.md`'s new changelog section.

## 2. Diagnosed and fixed the map/world coordinate mismatch

**Problem:** the robot's localization ("map" frame, built by driving the robot around with SLAM) and Gazebo's own absolute coordinate system ("world" frame) are two different coordinate systems that happen to both use meters, which looked like they should match but don't. The SLAM map's origin sits wherever the robot was first spawned — `(5.0, 6.6)`, hardcoded in the Gazebo launch file — not at world `(0, 0)`. So a pose read straight from `apartment.world`'s ground truth and sent to Nav2 as a "map frame" goal was off by that fixed offset (we verified this: an object's true world position was ~7m from where the mismatch put it; after correcting by the offset, it landed ~1m away — about right for a camera shot of a tabletop object).

**Fix:** added a static transform broadcaster (`world → map`, translation `(5.0, 6.6, 0)`, no rotation) to `hsrb_apartment_world.launch.py` *(official package — explicitly requested)*. This makes the conversion a standard TF lookup instead of manual arithmetic anywhere a world-frame pose needs to become a map-frame Nav2 goal.

**Verification commands:**
```bash
ros2 run tf2_ros tf2_echo world map              # should show (5.0, 6.6, 0.0)
ros2 run tf2_ros tf2_echo world base_footprint   # robot's Nav2-believed pose, in world frame
```
Compare the second one against the robot's actual position in the Gazebo GUI's Entity Tree to confirm localization is sane.

## 3. Restored the `fetcher` ROS2 package

It existed in an earlier commit (`c94d379`) and was accidentally deleted in the next one (`4472b76`). Restored `package.xml`, `setup.py`/`setup.cfg`, `resource/`, `test/`, and the five Python files (`gazebo_scene_graph.py`, `good_boy.py`, `robot_adapter.py`, `search_and_fetch_node.py`, `seeker.py`).

## 4. Fixed `robot_adapter.py` to use the world→map transform correctly

The scene graph (`build_scene_graph_gazebo.py`) reads object positions straight from `apartment.world`'s ground truth — i.e. **world frame**. But `navigate_to()` and `look_at()` were treating those positions as if they were already **map frame**, which is exactly the bug from §2. Fixed by stamping incoming poses/points as `frame_id="world"` and letting `tf2` convert them to `map` before use. `seeker.py` and `good_boy.py` needed no changes — neither ever touches world-frame ground truth (they compute positions live from the camera via TF, which was always self-consistent).

## 5. Fixed a wall-clock vs sim-time bug in `robot_adapter.py`

**Symptom:** `look_at()` and `grasp()` failed with `Lookup would require extrapolation into the future`.

**Cause:** the `HsrRobotAdapter` node never declared `use_sim_time`, so its own clock used wall-clock time (a huge Unix-epoch number) while Gazebo's TF tree is stamped with simulated time (small numbers, starting near 0). Every TF lookup looked like it was asking for a transform from the far future.

**Fix:** pass `parameter_overrides=[Parameter('use_sim_time', True)]` to the node's constructor so its clock matches the simulator's.

## 6. Replaced the custom IK-solver grasp path with AnyGrasp

**Why:** AnyGrasp's own grasp-pose selection already accounts for collisions, so running it through a *second*, separate collision-aware IK solver was redundant work and extra load for no benefit.

**Change:** `robot_adapter.py`'s `grasp()` now calls the existing `AnygraspClient` (`core/perception/grasping/anygrasp_client.py`, talking to the `docker/anygrasp` server) instead of the old `grasp_planner.compute_grasp_pose()` + `ik_solver_node/solve_ik_with_collision` service. Arm joints are computed with the same simple analytic formula already proven in `scripts/grasp_pipeline.py` (no IK service involved at all). The masked object's point cloud is reused as both the "item" and "env" cloud AnyGrasp's API expects (no separate background-scene capture yet — a possible future improvement if grasp quality needs it).

**Removed as a result:** `launch/ik_solver.launch.py` (dead), `core/perception/grasping/grasp_planner.py` (dead), and the now-unneeded `tmc_manipulation_msgs`/`moveit_msgs`/`tmc_ik_solver_node` entries in `fetcher/package.xml`.

## 7. Dead-code / unused-import sweep

Ran `flake8 --select=F401,F811,F841,F821` plus a manual cross-check of every file under `core/`, `scripts/`, `fetcher/`, `grasper/` against their entry points, and removed:
- `core/perception/scene_graph/build_scene_graph.py`: unused `LABEL_TO_ID` import
- `scripts/grasp_pipeline.py`: unused `CameraInfo`, `PoseStamped`, `tf2_geometry_msgs` imports
- `ros2_ws/src/fetcher/fetcher/good_boy.py`: unused `Node` import and an unused `feedback` variable (confirmed `nav.isTaskComplete()` already pumps the node — the discarded `getFeedback()` call wasn't doing anything)

Intentional standalone tools (`room_clustering.py`, `build_scene_graph.py`, everything in `scripts/`) were left alone — they're run directly (`python -m ...` / `python3 scripts/...`), not imported, so they only *look* unused to a naive "is this imported anywhere" check.

## 8. Simplification pass (reduce unnecessary complexity/duplication)

- **`scripts/sam3_segment.py`** (82 → ~32 lines): it hand-rolled the exact SAM3 wire-protocol parsing (the `X-Sam3-Meta` header, raw mask/box/score byte unpacking) that `core/perception/detection/sam3_client.py`'s `Sam3Client.detect()` already implements. Rewrote it to just use `Sam3Client`.
- **New file `core/llm_zone/_deepseek.py`**: factored out two pieces of logic that were copy-pasted three times across `scene_query.py` (twice) and `room_clustering.py` (once) — getting a DeepSeek `OpenAI` client (raising if `DEEPSEEK_API_KEY` isn't set) and stripping the `<think>...</think>` preamble that `deepseek-reasoner` prepends to its answers. Both files now import `get_client()`/`strip_think()` instead of repeating the logic.
- **`core/llm_zone/active_perception.py`'s `approach_pose()`**: it computed the approach yaw via `atan2(y - ay, x - ax)`, which *looked* like real geometry but, given how `ax`/`ay` are constructed two lines above, always evaluates to exactly `0`. Replaced with the literal constant `qz=0.0, qw=1.0` plus a comment explaining why, so a future reader isn't misled into thinking it's doing something direction-dependent.

All three were verified to produce identical output to the code they replaced before being committed to this log.

---

# Session 2 — 2026-06-18: AMCL pose-reset fix, networkx refactor, more cleanup

Continues the log above. Same scope note applies: everything below is inside `fetcher`/`core`/`scripts`/`docker`/`.devcontainer`, no official HSR/TMC packages touched.

## 9. Fixed AMCL snapping the robot back to the spawn point on every mission rerun

**Symptom:** run `ros2 run fetcher search_and_fetch "<object>"` once, the robot drives somewhere and finishes; run it again (Gazebo + Nav2 left running, only the mission script restarted) and it behaves as if the robot were back at the origin — bad nav goals, costmap built around the wrong pose.

**Cause:** `robot_adapter.py`'s `wait_until_ready()` called `self.nav.waitUntilNav2Active()`. `BasicNavigator`'s default `localizer='amcl'` makes that method publish a `(0, 0, 0)` `/initialpose` in a loop until it sees an `amcl_pose` reply (`nav2_simple_commander/robot_navigator.py:_waitForInitialPose`). Since a fresh `BasicNavigator()` is constructed every time the mission script runs, this fired every run — forcibly resetting AMCL's already-correct belief to the spawn point, regardless of where the robot actually was.

**Fix:** `wait_until_ready()` now calls `self.nav.waitUntilNav2Active(localizer='bt_navigator')` instead. Passing a non-`'amcl'` value skips the `/initialpose` publish entirely while still blocking until Nav2 is active (bt_navigator being active already implies amcl is too, since it starts earlier in the same bringup).

**File:** `ros2_ws/src/fetcher/fetcher/robot_adapter.py`

## 10. Second simplification pass + SceneGraph internals → networkx

- **Deleted `scripts/grasp_pipeline.py`** (375 lines): a standalone prototype, never imported anywhere, that hand-rolled the AnyGrasp REST call, `camera_to_base`, and `grasp_pose_to_joints` — all three already live properly in `robot_adapter.py` + `core/perception/grasping/anygrasp_client.py` (which the prototype predates). It was the throwaway script that became the real implementation; keeping it around was just a second, divergence-prone copy of the grasp math.
- **`scripts/sam3_segment.py` and `scripts/sam3_live_detect.py`** each had their own mask/box overlay-drawing code, subtly different (one blended in-place, the other used `cv2.addWeighted`). Extracted a single `draw_detections()` into `core/perception/detection/sam3_client.py`; both scripts now call it.
- **`build_scene_graph.py` and `build_scene_graph_gazebo.py`** duplicated their "finalize" tail verbatim: wire up `update_connection`, rebuild the KD-tree, assign colors, print a per-node summary. Moved this into `SceneGraph.finalize_and_report(source_label)`; both builders now make one call instead.
- **`DATA_DIR`/`GRAPH_DIR`** (`Path(os.environ.get("HRL_DATA_DIR", ...)) / "scene_graph"`) was copy-pasted identically in `scene_query.py`, `room_clustering.py`, and both scene-graph builders. Moved into `core/utils/config.py`, imported everywhere else.
- **Moved `active_perception.py`** from `core/llm_zone/` to `core/perception/` — it's pure nav geometry (`approach_pose`, `viewpoints`), not LLM-calling code, so it didn't belong next to `scene_query.py`/`room_clustering.py`/`_deepseek.py`.
- **`SceneGraph`'s manual `outgoing`/`ingoing` dict pair → `networkx.DiGraph`.** The class kept two dicts (`movable_id → furniture_id` and `furniture_id → [movable_ids]`) in sync by hand in `update_connection()` and `remove_node()` — exactly the kind of dual-bookkeeping that drifts out of sync. Replaced with a single `self.graph = nx.DiGraph()`; `update_connection()` and `remove_node()` now just call `add_edge`/`remove_node`, which networkx keeps consistent atomically. The on-disk `graph.json`/`scene.json` schema (what `scene_query.py` reads) is unchanged. Added `networkx` to `.devcontainer/Dockerfile`.
  - Caught two real bugs while verifying this with a synthetic add/connect/re-score/remove/save round-trip before trusting it: `graph.out_edges(id)` raises instead of returning empty when `id` was never added as a node (fixed with an `if id in self.graph` guard), and `dict(graph.edges())` does *not* do what it looks like — `EdgeView` implements the `Mapping` protocol, so `dict()` treats edge tuples as keys via `.keys()`/`__getitem__` instead of treating them as `(u, v)` pairs. Fixed with an explicit `{u: v for u, v in self.graph.edges()}`.
- **`ros2_ws/src/fetcher/setup.py`**: removed the `good_boy`/`seeker` console_script entries after those two files were deleted (they were early standalone Nav2/YOLO test scripts, superseded by `robot_adapter.py` + `search_and_fetch.py`). Without this, `colcon build` fails looking for the now-missing modules.

All of the above were verified by importing/byte-compiling every touched file and exercising the new `SceneGraph` logic directly (not just by inspection) before being written down here.

---

# Session 3 — 2026-06-19: live ground-truth poses, Nav2 "unconnected trees" diagnosis

## 11. Added a live pose snapshot to `build_scene_graph_gazebo.py`

**Why:** the script read every object's pose once, from the world file's static spawn `<pose>`. Anything that moved after spawn (robot bumps it, gets picked up) made the ground-truth scene graph stale for the rest of the run.

**Change:** added `fetch_live_poses()`, which grabs one snapshot of every model's current world-frame pose from Gazebo's own `/world/default/pose/info` topic — bridged to ROS2 as `tf2_msgs/msg/TFMessage` via a new line in `gz_parameter_bridge_node()` (`gazebo_bringup.launch.py`). `load_registry()` now uses a live pose per object when one's available, falling back to the world file's static spawn pose (so the script still works with Gazebo not running, e.g. for offline testing). `gazebo_geometry.py` gained `quat_to_euler()` (inverse of the existing `euler_to_matrix()`) to convert the bridged quaternion. A `--static-pose` flag forces the old spawn-pose-only behavior.

**Files:** `core/perception/scene_graph/build_scene_graph_gazebo.py`, `core/perception/scene_graph/gazebo_geometry.py`, `ros2_ws/src/hsrb_simulator/hsrb_gazebo_bringup/launch/gazebo_bringup.launch.py` *(official package — one new bridge line, no existing bridges touched)*

**Verification:** ran `fetch_live_poses()` standalone with no Gazebo running — confirmed it times out after ~2s and returns `{}` (graceful fallback) instead of hanging. The live-bridge path itself needs a real run (relaunch the sim, move an object, rerun the builder) to visually confirm the snapshot tracks it.

## 12. Diagnosed a `navigation_launch.py` standalone "two unconnected trees" error

**Symptom:** running `ros2 launch hsrb_rosnav_config navigation_launch.py map:=map.yaml ...` by hand failed two ways in sequence: `map_server` rejected `map.yaml` (wrong filename — `ipad_map.yaml` is what actually exists at the repo root, and it's for a different, real-world space anyway), and even after pointing at a real map, `planner_server` kept reporting `"map" and "base_link" ... not part of the same tree"`.

**Cause:** launching `navigation_launch.py` alone never gets a `world → map` transform — that's published by `world_to_map_tf` inside `hsrb_apartment_world.launch.py` (added in §2 above), which also spawns the robot in the matching world at the exact pose the map was built from. Skip that launch file and `map` has no parent, so it can never connect to the robot's own TF tree (rooted at `world`).

**Fix:** no code change needed — this repo's documented pipeline (see "Current pipeline" below) already covers it correctly with the matching `apartment_world_map.yaml`. Pointed at it directly, plus the equivalent one-command alternative: `ros2 launch hsrb_gazebo_launch hsrb_apartment_world.launch.py use_navigation:=true` (wires world + map + robot + nav together in one launch instead of the two-step manual flow).

## Current pipeline — how to run it end to end

```bash
# 1. Sim (world→map static transform is now baked into this launch file)
ros2 launch hsrb_gazebo_launch hsrb_apartment_world.launch.py use_navigation:=False

# 2. Navigation, pointed at the SLAM-built map
ros2 launch hsrb_rosnav_config navigation_launch.py \
  map:=/home/ws/apartment_world_map.yaml use_sim_time:=True initial_orientation_xyzw:=[0,0,0,1]

# 3. Perception servers (separate terminals/machines)
bash docker/sam3/run_sam3.sh
bash docker/anygrasp/run_anygrasp.sh

# 4. Build the scene graph from world ground truth (run once per world)
ros2 run fetcher gazebo_scene_graph

# 5. Run the mission
ros2 run fetcher search_and_fetch "pringles can"
```

No `ik_solver.launch.py` step anymore — AnyGrasp replaced that whole path.

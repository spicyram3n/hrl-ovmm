#!/usr/bin/env python3
"""
ros2 run fetcher gazebo_scene_graph

Builds the 3D scene graph from the apartment.world ground truth (object
names, poses and sizes) and saves graph.json / scene.json / objects/*.json
to $HRL_DATA_DIR/scene_graph.
"""

from core.perception.scene_graph.build_scene_graph_gazebo import main

if __name__ == '__main__':
    main()

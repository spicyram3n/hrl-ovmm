"""
head_geometry.py
-----------------
Pure geometry for pointing the HSR head camera at a 3D point (active
perception / "look_at"). Nothing here talks to ROS or TF — the caller
(ros2_ws/src/fetcher/fetcher/robot_adapter.py) does the TF lookups and
hands this module plain (x, y, z) floats.

head_pan_joint:  axis +Z, limits [-3.84, 1.75] rad
head_tilt_joint: axis -Y, limits [-1.57, 0.52] rad
(see ros2_ws/src/hsrb_common/hsrc_description/urdf/head_v0/head.urdf.xacro)
"""

import math

PAN_LIMITS = (-3.84, 1.75)
TILT_LIMITS = (-1.57, 0.52)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def pan_delta(x_in_pan_frame: float, y_in_pan_frame: float) -> float:
    """
    Additional head_pan_joint rotation (rad) so the pan link's local +X
    axis points at (x, y), given in the CURRENT head_pan_link frame.

    Add this to the current head_pan_joint reading to get the new target.
    """
    return math.atan2(y_in_pan_frame, x_in_pan_frame)


def tilt_for_target(x_base: float, y_base: float, z_base: float, head_height: float) -> float:
    """
    head_tilt_joint angle (rad) to point the camera at a target given in
    base_footprint frame. head_height = height of head_pan_link above
    base_footprint (read from TF).

    Negative tilt = look down (per seeker.py's HEAD_TILT=-0.3 "slightly
    down" convention and the tilt joint's -Y axis).
    """
    horiz = math.hypot(x_base, y_base)
    return math.atan2(z_base - head_height, max(horiz, 1e-3))

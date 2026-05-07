"""
Coordinate transform and pose utilities shared across simulation and hardware.

Functions here were previously duplicated in env_setup.py and mininav/interface.py.
"""

import math
from typing import Any, List, Tuple

import numpy as np


def transform_to_global(
    x: float, y: float, initial_yaw: float
) -> Tuple[float, float]:
    """Transform relative x,y to global frame given initial yaw (degrees)."""
    theta = np.radians(initial_yaw)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    global_x = x * cos_theta - y * sin_theta
    global_y = x * sin_theta + y * cos_theta
    return global_x, global_y


def normalize_angle(angle: float) -> float:
    """Normalize angle to (-180, 180] degrees."""
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def normalize_initial_pos(initial_pos: List[float]) -> List[float]:
    """Validate and return initial_pos as [x, y, z, yaw].

    Accepts 4 elements [x,y,z,yaw] directly, or legacy 5+ element
    formats [x,y,z,_,yaw] (index 3 is ignored, yaw taken from index 4).
    Always returns a 4-element list.
    """
    if len(initial_pos) >= 5:
        return [float(initial_pos[0]), float(initial_pos[1]),
                float(initial_pos[2]), float(initial_pos[4])]
    if len(initial_pos) == 4:
        return [float(x) for x in initial_pos]
    raise ValueError("initial_pos must have at least 4 elements (x,y,z,yaw)")


def relative_pose_to_world(
    origin_x: float,
    origin_y: float,
    origin_z: float,
    origin_yaw: float,
    relative_pose: List[float],
) -> Tuple[float, float, float, float]:
    """Convert relative pose [x, y, z, yaw_deg] to world (x, y, z, yaw_deg) given origin."""
    rx, ry, rz, yaw_deg = (relative_pose[0], relative_pose[1],
                            relative_pose[2], relative_pose[3])
    gx, gy = transform_to_global(rx, ry, origin_yaw)
    world_x = origin_x + gx
    world_y = origin_y + gy
    world_z = origin_z + rz
    world_yaw = normalize_angle(yaw_deg + origin_yaw)
    return (world_x, world_y, world_z, world_yaw)


def relative_pose(current_world: List[float], origin_world: List[float]) -> List[float]:
    """Compute relative pose [dx, dy, dz, dyaw] from world-frame poses."""
    return [
        float(current_world[0] - origin_world[0]),
        float(current_world[1] - origin_world[1]),
        float(current_world[2] - origin_world[2]),
        float(normalize_angle(current_world[3] - origin_world[3])),
    ]


def reframe_openvla_action_to_subgoal(
    action_poses: List[Any],
    current_pose: List[float],
    openvla_pose_origin: List[float],
) -> List[Any]:
    """Reframe OpenVLA-server action poses into the subgoal-relative frame.

    Once a correction has rebased ``openvla_pose_origin`` off zero, the proprio
    sent to the OpenVLA server is ``current_pose - openvla_pose_origin``
    (translated, with origin yaw subtracted). The server then rotates the
    egocentric raw action by the proprio yaw,
    ``current_pose[3] - openvla_pose_origin[3]``, which under-rotates by
    ``openvla_pose_origin[3]`` -- the yaw the drone was holding when the
    corrective was issued. The returned ``(x, y)`` therefore lives in a frame
    rotated by ``-openvla_pose_origin[3]`` relative to the subgoal frame.

    This helper applies the missing rotation and translation so dx/dy align
    with the drone's heading at the start of the low-level instruction
    (not the subgoal start).

    Conventions:
      - ``current_pose`` and ``openvla_pose_origin`` are
        ``[x, y, z, yaw_deg]`` in subgoal-relative coordinates.
      - Each ``action_pose`` is ``[x, y, z, yaw_rad]`` (server output convention,
        yaw in radians) in proprio-relative coordinates.
      - Returned poses use the same shape; yaw stays in radians.
      - When ``openvla_pose_origin`` is all-zero (no correction yet), the
        returned list is the same object as ``action_poses`` unchanged.
      - Non-pose entries (anything that is not a length>=4 list/tuple) are
        passed through verbatim.
    """
    if not any(o != 0.0 for o in openvla_pose_origin):
        return action_poses
    yaw_origin_deg = float(openvla_pose_origin[3])
    yaw_origin_rad = math.radians(yaw_origin_deg)
    proprio_x = current_pose[0] - openvla_pose_origin[0]
    proprio_y = current_pose[1] - openvla_pose_origin[1]
    cos_o = math.cos(yaw_origin_rad)
    sin_o = math.sin(yaw_origin_rad)
    out: List[Any] = []
    for pose in action_poses:
        if isinstance(pose, (list, tuple)) and len(pose) >= 4:
            delta_x = float(pose[0]) - proprio_x
            delta_y = float(pose[1]) - proprio_y
            rotated_dx = cos_o * delta_x - sin_o * delta_y
            rotated_dy = sin_o * delta_x + cos_o * delta_y
            out.append([
                current_pose[0] + rotated_dx,
                current_pose[1] + rotated_dy,
                float(pose[2]) + openvla_pose_origin[2],
                float(pose[3]) + yaw_origin_rad,
            ])
        else:
            out.append(pose)
    return out


def parse_position(s: str) -> List[float]:
    """Parse comma-separated 'x,y,z,yaw' string into 4 floats. Supports negative numbers."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            "Position must be 4 comma-separated numbers: x,y,z,yaw "
            "(e.g. -600,-1270,128,61)"
        )
    return [float(x) for x in parts]

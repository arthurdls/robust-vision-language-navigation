"""
Pose calculation utilities for Unreal Engine coordinate transforms.

Vendored from UAV-Flow/UAV-Flow-Eval/relative.py (commit 0114801).
"""

import numpy as np


def calculate_new_pose(location, rotation, backward_distance, up_distance):
    """Calculate a new pose based on an object's location and rotation.

    Args:
        location: [x, y, z] position of the object.
        rotation: [pitch, yaw, roll] rotation of the object in degrees.
        backward_distance: Distance to move backward.
        up_distance: Distance to move upward.
    Returns:
        Tuple of (new_location, new_rotation).
    """
    pitch, yaw, roll = np.radians(rotation)

    forward_vector = np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ])

    up_vector = np.array([0, 0, 1])

    new_location = np.array(location) - backward_distance * forward_vector + up_distance * up_vector
    new_rotation = rotation

    return new_location.tolist(), new_rotation

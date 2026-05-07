"""Tests for rvln.sim.transforms -- coordinate transform utilities."""

import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from rvln.sim.transforms import (
    normalize_angle,
    normalize_initial_pos,
    parse_position,
    reframe_openvla_action_to_subgoal,
    relative_pose,
    relative_pose_to_world,
    transform_to_global,
)


class TestNormalizeAngle:
    def test_zero(self):
        assert normalize_angle(0) == 0

    def test_positive_in_range(self):
        assert normalize_angle(90) == 90

    def test_negative_in_range(self):
        assert normalize_angle(-90) == -90

    def test_wrap_360(self):
        assert normalize_angle(360) == 0

    def test_wrap_270(self):
        assert normalize_angle(270) == -90

    def test_wrap_540(self):
        assert normalize_angle(540) == 180

    def test_negative_wrap(self):
        assert normalize_angle(-270) == 90


class TestParsePosition:
    def test_basic(self):
        assert parse_position("-600,-1270,128,61") == [-600.0, -1270.0, 128.0, 61.0]

    def test_with_spaces(self):
        assert parse_position(" 1 , 2 , 3 , 4 ") == [1.0, 2.0, 3.0, 4.0]

    def test_too_few_values(self):
        with pytest.raises(ValueError, match="4 comma-separated"):
            parse_position("1,2,3")

    def test_too_many_values(self):
        with pytest.raises(ValueError, match="4 comma-separated"):
            parse_position("1,2,3,4,5")


class TestNormalizeInitialPos:
    def test_five_elements_extracts_yaw_from_idx4(self):
        result = normalize_initial_pos([1, 2, 3, 0, 90])
        assert result == [1.0, 2.0, 3.0, 90.0]

    def test_four_elements_passthrough(self):
        result = normalize_initial_pos([1, 2, 3, 90])
        assert result == [1.0, 2.0, 3.0, 90.0]

    def test_three_elements_raises(self):
        with pytest.raises(ValueError):
            normalize_initial_pos([1, 2, 3])


class TestTransformToGlobal:
    def test_zero_yaw(self):
        gx, gy = transform_to_global(10.0, 0.0, 0.0)
        assert abs(gx - 10.0) < 1e-6
        assert abs(gy - 0.0) < 1e-6

    def test_90_degree_yaw(self):
        gx, gy = transform_to_global(10.0, 0.0, 90.0)
        assert abs(gx - 0.0) < 1e-6
        assert abs(gy - 10.0) < 1e-6


class TestRelativePose:
    def test_same_position(self):
        result = relative_pose([10, 20, 30, 45], [10, 20, 30, 45])
        assert all(abs(v) < 1e-6 for v in result)

    def test_offset(self):
        result = relative_pose([15, 25, 35, 50], [10, 20, 30, 45])
        assert abs(result[0] - 5.0) < 1e-6
        assert abs(result[1] - 5.0) < 1e-6
        assert abs(result[2] - 5.0) < 1e-6
        assert abs(result[3] - 5.0) < 1e-6

    def test_angle_wrapping(self):
        result = relative_pose([0, 0, 0, 350], [0, 0, 0, 10])
        assert abs(result[3] - (-20.0)) < 1e-6


class TestRelativePoseToWorld:
    def test_identity(self):
        wx, wy, wz, wyaw = relative_pose_to_world(100, 200, 50, 0, [0, 0, 0, 0])
        assert abs(wx - 100) < 1e-6
        assert abs(wy - 200) < 1e-6
        assert abs(wz - 50) < 1e-6
        assert abs(wyaw - 0) < 1e-6

    def test_forward_movement(self):
        wx, wy, wz, wyaw = relative_pose_to_world(0, 0, 0, 0, [10, 0, 0, 0])
        assert abs(wx - 10) < 1e-6
        assert abs(wy - 0) < 1e-6


class TestReframeOpenvlaActionToSubgoal:
    """Tests for the post-correction action-pose reframing helper.

    Convention reminder:
      - current_pose / openvla_pose_origin: [x, y, z, yaw_deg], subgoal frame.
      - action_pose entries: [x, y, z, yaw_rad], proprio frame
        (proprio_pos = current_pose - openvla_pose_origin element-wise;
         proprio_yaw = current_pose[3] - openvla_pose_origin[3]).
    """

    def test_no_correction_is_passthrough(self):
        # openvla_pose_origin is all zeros (no correction has fired yet),
        # so the helper must return action_poses unchanged.
        action_poses = [[1.0, 2.0, 3.0, 0.5]]
        current = [4.0, 5.0, 6.0, 30.0]
        origin = [0.0, 0.0, 0.0, 0.0]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        assert out is action_poses

    def test_right_turn_then_forward_moves_in_minus_y(self):
        # Bug repro from the original report: drone turned right 90 degrees
        # within the subgoal, corrective fires "move forward". The proprio
        # sent to the server is (0,0,0,0); raw forward action (1,0,0,0)
        # comes back from the server unrotated. The reframer must rotate
        # by -90 deg (origin yaw) so dx/dy align with the drone's actual
        # heading: forward should be -y in the subgoal frame, not +x.
        # OLD buggy behavior produced (1, 0); the fix produces (0, -1).
        origin = [0.0, 0.0, 0.0, -90.0]
        current = list(origin)  # no movement yet within corrective
        action_poses = [[1.0, 0.0, 0.0, 0.0]]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        assert len(out) == 1
        x, y, z, yaw_rad = out[0]
        assert abs(x - 0.0) < 1e-9
        assert abs(y - (-1.0)) < 1e-9
        assert abs(z - 0.0) < 1e-9
        assert abs(yaw_rad - math.radians(-90.0)) < 1e-9

    def test_left_turn_then_forward_moves_in_plus_y(self):
        # Mirror of the right-turn case: +90 deg origin yaw, forward becomes +y.
        origin = [0.0, 0.0, 0.0, 90.0]
        current = list(origin)
        action_poses = [[1.0, 0.0, 0.0, 0.0]]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        x, y, _z, _yaw = out[0]
        assert abs(x - 0.0) < 1e-9
        assert abs(y - 1.0) < 1e-9

    def test_180_turn_flips_forward_to_minus_x(self):
        # 180 deg origin yaw: forward action (1,0) ends up at -x.
        origin = [0.0, 0.0, 0.0, 180.0]
        current = list(origin)
        action_poses = [[1.0, 0.0, 0.0, 0.0]]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        x, y, _z, _yaw = out[0]
        assert abs(x - (-1.0)) < 1e-9
        assert abs(y - 0.0) < 1e-9

    def test_movement_within_corrective_is_handled(self):
        # Drone has moved 0.4m forward (subgoal -y because origin yaw=-90)
        # since the corrective rebased the origin. proprio_pos is therefore
        # (0, -0.4) in subgoal axes; raw forward of an additional 1m means
        # the server returns proprio_pos + R(0)*(1,0) = (1, -0.4).
        # Expected subgoal-frame target: (0, -0.4 - 1) = (0, -1.4).
        origin = [0.0, 0.0, 0.0, -90.0]
        current = [0.0, -0.4, 0.0, -90.0]
        action_poses = [[1.0, -0.4, 0.0, 0.0]]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        x, y, _z, _yaw = out[0]
        assert abs(x - 0.0) < 1e-9
        assert abs(y - (-1.4)) < 1e-9

    def test_yaw_is_offset_by_origin_yaw_in_radians(self):
        # The server returns yaw in radians, openvla_pose_origin yaw is in
        # degrees. The helper must add math.radians(origin_yaw_deg).
        origin = [0.0, 0.0, 0.0, 45.0]
        current = list(origin)
        action_poses = [[0.0, 0.0, 0.0, math.radians(20.0)]]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        _x, _y, _z, yaw_rad = out[0]
        assert abs(yaw_rad - math.radians(65.0)) < 1e-9

    def test_z_translates_without_rotation(self):
        # z is purely additive (origin_z added back); it does not rotate.
        origin = [0.0, 0.0, 1.5, 90.0]
        current = list(origin)
        action_poses = [[0.0, 0.0, 0.7, 0.0]]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        _x, _y, z, _yaw = out[0]
        assert abs(z - 2.2) < 1e-9

    def test_origin_translation_is_added(self):
        # Pure translation (origin xy non-zero, yaw 0): output is server
        # pose plus origin xy. Note: origin yaw must be non-zero or all-zero
        # short-circuits, so we use a tiny non-zero yaw and check xy.
        origin = [3.0, 4.0, 0.0, 0.0001]
        current = list(origin)
        # Server output that, after rotating by ~0 deg, produces (1, 0):
        action_poses = [[1.0, 0.0, 0.0, 0.0]]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        x, y, _z, _yaw = out[0]
        # delta = (1,0) - proprio(0,0) = (1,0); rotate by ~0 deg -> ~(1,0);
        # add current_pose (3, 4) -> (~4, ~4).
        assert abs(x - 4.0) < 1e-3
        assert abs(y - 4.0) < 1e-3

    def test_non_pose_entries_pass_through(self):
        # The helper preserves anything that isn't a length-4 list/tuple.
        origin = [0.0, 0.0, 0.0, -90.0]
        current = list(origin)
        action_poses = [
            [1.0, 0.0, 0.0, 0.0],
            "not a pose",
            [1.0, 2.0],  # too short
        ]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        assert len(out) == 3
        # First was reframed:
        assert abs(out[0][0] - 0.0) < 1e-9
        assert abs(out[0][1] - (-1.0)) < 1e-9
        # Second and third pass through verbatim:
        assert out[1] == "not a pose"
        assert out[2] == [1.0, 2.0]

    def test_multiple_poses_each_reframed(self):
        # A list of pose targets all get reframed consistently.
        origin = [0.0, 0.0, 0.0, -90.0]
        current = list(origin)
        action_poses = [
            [1.0, 0.0, 0.0, 0.0],   # forward 1m -> (0, -1)
            [2.0, 0.0, 0.0, 0.0],   # forward 2m -> (0, -2)
            [0.0, 1.0, 0.0, 0.0],   # left 1m in egocentric (server-rotated 0) -> (1, 0)
        ]
        out = reframe_openvla_action_to_subgoal(action_poses, current, origin)
        assert abs(out[0][0] - 0.0) < 1e-9 and abs(out[0][1] - (-1.0)) < 1e-9
        assert abs(out[1][0] - 0.0) < 1e-9 and abs(out[1][1] - (-2.0)) < 1e-9
        assert abs(out[2][0] - 1.0) < 1e-9 and abs(out[2][1] - 0.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

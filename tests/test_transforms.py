"""Tests for rvln.sim.transforms -- coordinate transform utilities."""

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
    def test_five_elements_passthrough(self):
        result = normalize_initial_pos([1, 2, 3, 0, 90])
        assert result == [1, 2, 3, 0, 90]

    def test_four_elements_expand(self):
        result = normalize_initial_pos([1, 2, 3, 90])
        assert result == [1.0, 2.0, 3.0, 0.0, 90.0]

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

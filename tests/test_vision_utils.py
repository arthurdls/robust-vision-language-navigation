"""
Unit tests for vision utilities: sample_frames_every_n.

No API calls, no GPU. Tests the frame sampling logic that is used in every
checkpoint and convergence call across all monitored conditions.

Run: conda run -n rvln-sim pytest tests/test_vision_utils.py -v
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from rvln.ai.utils.vision import sample_frames_every_n


class TestSampleFramesEveryN:
    def test_empty_input(self):
        assert sample_frames_every_n([], 10) == []

    def test_single_frame(self):
        frames = ["a"]
        assert sample_frames_every_n(frames, 10) == ["a"]

    def test_n_equals_1_returns_all(self):
        frames = list(range(5))
        assert sample_frames_every_n(frames, 1) == [0, 1, 2, 3, 4]

    def test_n_greater_than_frame_count(self):
        frames = ["a", "b", "c"]
        result = sample_frames_every_n(frames, 100)
        # Should return at least the last frame
        assert "c" in result
        assert len(result) >= 1

    def test_exact_multiple(self):
        """10 frames, n=5 -> should include first (0) and last (9)."""
        frames = list(range(10))
        result = sample_frames_every_n(frames, 5)
        assert frames[-1] in result
        assert len(result) >= 2

    def test_typical_case(self):
        """144 frames, n=10 -> docs say indices [3, 13, 23, ..., 133, 143]."""
        frames = list(range(144))
        result = sample_frames_every_n(frames, 10)
        assert result[-1] == 143
        assert result[0] == 3
        assert len(result) == 15

    def test_preserves_order(self):
        frames = list(range(20))
        result = sample_frames_every_n(frames, 5)
        assert result == sorted(result)

    def test_always_includes_last_frame(self):
        for total in [1, 5, 10, 20, 100]:
            frames = list(range(total))
            result = sample_frames_every_n(frames, 7)
            assert result[-1] == total - 1

    def test_n_equals_frame_count(self):
        frames = list(range(10))
        result = sample_frames_every_n(frames, 10)
        assert frames[-1] in result
        assert len(result) >= 1

    def test_two_frames_n_1(self):
        frames = ["first", "second"]
        result = sample_frames_every_n(frames, 1)
        assert result == ["first", "second"]

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            sample_frames_every_n([1, 2, 3], 0)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            sample_frames_every_n([1, 2, 3], -1)

    def test_works_with_path_objects(self):
        frames = [Path(f"/tmp/frame_{i}.png") for i in range(20)]
        result = sample_frames_every_n(frames, 5)
        assert all(isinstance(p, Path) for p in result)
        assert result[-1] == Path("/tmp/frame_19.png")

    def test_no_duplicates(self):
        for n in [1, 2, 3, 5, 10, 50]:
            frames = list(range(50))
            result = sample_frames_every_n(frames, n)
            assert len(result) == len(set(result)), f"Duplicates found with n={n}"

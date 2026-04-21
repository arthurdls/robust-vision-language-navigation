from rvln.ai.diary_monitor import LiveDiaryMonitor


def test_no_stall_when_not_enough_history():
    """Stall detection needs at least stall_window checkpoints."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.1, 0.12]
    m._stall_window = 3
    m._stall_threshold = 0.05
    assert m._is_stalled() is False


def test_stall_detected_when_flat():
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.30, 0.31, 0.32]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is True


def test_no_stall_when_progressing():
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.30, 0.40, 0.50]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is False


def test_no_stall_when_completion_high():
    """Don't ask for help if already nearly done."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.85, 0.86, 0.86]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is False


def test_stall_only_looks_at_last_window():
    """Earlier progress doesn't mask a recent plateau."""
    m = LiveDiaryMonitor.__new__(LiveDiaryMonitor)
    m._completion_history = [0.10, 0.20, 0.30, 0.31, 0.32]
    m._stall_window = 3
    m._stall_threshold = 0.05
    m._stall_completion_floor = 0.8
    assert m._is_stalled() is True

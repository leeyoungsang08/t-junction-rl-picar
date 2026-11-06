import pytest

from src.service_guard import DirectionServiceGuard


def test_oldest_backlog_wins_and_other_direction_keeps_its_age():
    guard = DirectionServiceGuard(3)
    for i in range(5):
        # E is waiting to be inserted; S starts later, with a longer queue.
        guard.observe_second(0, "green", [0, 0, 8 if i >= 2 else 0], [0, 1, 0])
    assert guard.ages == (0, 5, 3)
    assert guard.choose(0, [True, True, True], 0, False, [0, 0, 8], [0, 1, 0]) == (1, 1)
    guard.observe_second(1, "green", [0, 1, 8], [0, 1, 0])
    assert guard.ages == (0, 0, 4)
    assert guard.choose(1, [True, True, True], 1, False, [0, 1, 8], [0, 1, 0]) == (2, 2)


def test_clearance_and_minimum_green_still_gate_priority():
    guard = DirectionServiceGuard(2)
    guard.observe_second(0, "yellow", [2, 3, 0], [0, 0, 0])
    guard.observe_second(0, "all_red", [2, 3, 0], [0, 0, 0])
    assert guard.ages == (2, 2, 0)
    assert guard.choose(0, [True, False, False], 0, True, [2, 3, 0], [0, 0, 0]) == (0, None)
    assert guard.choose(0, [True, False, False], 0, False, [2, 3, 0], [0, 0, 0]) == (0, None)
    assert guard.choose(0, [True, True, True], 0, True, [2, 3, 0], [0, 0, 0]) == (0, None)
    assert guard.choose(2, [False, True, True], 0, False, [2, 3, 0], [0, 0, 0]) == (1, 1)


def test_no_stale_override_after_demand_clears_and_at_reset():
    guard = DirectionServiceGuard(1)
    guard.observe_second(0, "green", [0, 1, 0], [0, 0, 0])
    assert guard.choose(0, [True, True, True], 0, False, [0, 0, 0], [0, 0, 0]) == (0, None)
    guard.observe_second(0, "green", [0, 0, 0], [0, 0, 0])
    assert guard.ages == (0, 0, 0)
    guard.observe_second(0, "green", [0, 1, 0], [0, 0, 0])
    guard.reset()
    assert guard.ages == (0, 0, 0)
    with pytest.raises(ValueError):
        guard.choose(1, [True, False, False], 0, False, [0, 1, 0], [0, 0, 0])
